//===--- Types.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Types.h"
#include "TypesInternal.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/FileEntry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

namespace clang::include_cleaner {

std::string Symbol::name() const {
  switch (kind()) {
  case include_cleaner::Symbol::Macro:
    return macro().Name->getName().str();
  case include_cleaner::Symbol::Declaration:
    return llvm::dyn_cast<NamedDecl>(&declaration())
        ->getQualifiedNameAsString();
  }
  llvm_unreachable("Unknown symbol kind");
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Symbol &S) {
  switch (S.kind()) {
  case Symbol::Declaration:
    if (const auto *ND = llvm::dyn_cast<NamedDecl>(&S.declaration()))
      return OS << ND->getQualifiedNameAsString();
    return OS << S.declaration().getDeclKindName();
  case Symbol::Macro:
    return OS << S.macro().Name->getName();
  }
  llvm_unreachable("Unhandled Symbol kind");
}

llvm::StringRef Header::resolvedPath() const {
  switch (kind()) {
  case include_cleaner::Header::Physical:
    return physical().getName();
  case include_cleaner::Header::Standard:
    return standard().name().trim("<>\"");
  case include_cleaner::Header::Verbatim:
    return verbatim().trim("<>\"");
  }
  llvm_unreachable("Unknown header kind");
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Header &H) {
  switch (H.kind()) {
  case Header::Physical:
    return OS << H.physical().getName();
  case Header::Standard:
    return OS << H.standard().name();
  case Header::Verbatim:
    return OS << H.verbatim();
  }
  llvm_unreachable("Unhandled Header kind");
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Include &I) {
  return OS << I.Line << ": " << I.quote() << " => "
            << (I.Resolved ? I.Resolved->getName() : "<missing>");
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SymbolReference &R) {
  // We can't decode the Location without SourceManager. Its raw representation
  // isn't completely useless (and distinguishes SymbolReference from Symbol).
  return OS << R.RT << " reference to " << R.Target << "@0x"
            << llvm::utohexstr(
                   R.RefLocation.getRawEncoding(), /*LowerCase=*/false,
                   /*Width=*/CHAR_BIT * sizeof(SourceLocation::UIntTy));
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, RefType T) {
  switch (T) {
  case RefType::Explicit:
    return OS << "explicit";
  case RefType::Implicit:
    return OS << "implicit";
  case RefType::Ambiguous:
    return OS << "ambiguous";
  }
  llvm_unreachable("Unexpected RefType");
}

std::string Include::quote() const {
  return (llvm::StringRef(Angled ? "<" : "\"") + Spelled +
          (Angled ? ">" : "\""))
      .str();
}

static llvm::SmallString<128> normalizePath(llvm::StringRef Path) {
  namespace path = llvm::sys::path;

  llvm::SmallString<128> P = Path;
  path::remove_dots(P, /*remove_dot_dot=*/true);
  path::native(P, path::Style::posix);
  while (!P.empty() && P.back() == '/')
    P.pop_back();
  return P;
}

void Includes::addSearchDirectory(llvm::StringRef Path) {
  SearchPath.try_emplace(normalizePath(Path));
}

void Includes::add(const Include &I) {
  namespace path = llvm::sys::path;

  unsigned Index = All.size();
  All.push_back(I);
  auto BySpellingIt = BySpelling.try_emplace(I.Spelled).first;
  All.back().Spelled = BySpellingIt->first(); // Now we own the backing string.

  BySpellingIt->second.push_back(Index);
  ByLine[I.Line] = Index;

  if (!I.Resolved)
    return;
  ByFile[&I.Resolved->getFileEntry()].push_back(Index);

  // While verbatim headers ideally should match #include spelling exactly,
  // we want to be tolerant of different spellings of the same file.
  //
  // If the search path includes "/a/b" and "/a/b/c/d",
  // verbatim "e/f" should match (spelled=c/d/e/f, resolved=/a/b/c/d/e/f).
  // We assume entry's (normalized) name will match the search dirs.
  auto Path = normalizePath(I.Resolved->getName());
  for (llvm::StringRef Parent = path::parent_path(Path); !Parent.empty();
       Parent = path::parent_path(Parent)) {
    if (!SearchPath.contains(Parent))
      continue;
    llvm::StringRef Rel =
        llvm::StringRef(Path).drop_front(Parent.size()).ltrim('/');
    BySpellingAlternate[Rel].push_back(Index);
  }
}

const Include *Includes::atLine(unsigned OneBasedIndex) const {
  auto It = ByLine.find(OneBasedIndex);
  return (It == ByLine.end()) ? nullptr : &All[It->second];
}

llvm::SmallVector<const Include *> Includes::match(Header H) const {
  llvm::SmallVector<const Include *> Result;
  switch (H.kind()) {
  case Header::Physical:
    for (unsigned I : ByFile.lookup(H.physical()))
      Result.push_back(&All[I]);
    break;
  case Header::Standard:
    for (unsigned I : BySpelling.lookup(H.standard().name().trim("<>")))
      Result.push_back(&All[I]);
    break;
  case Header::Verbatim: {
    llvm::StringRef Spelling = H.verbatim().trim("\"<>");
    for (unsigned I : BySpelling.lookup(Spelling))
      Result.push_back(&All[I]);
    for (unsigned I : BySpellingAlternate.lookup(Spelling))
      if (!llvm::is_contained(Result, &All[I]))
        Result.push_back(&All[I]);
    break;
  }
  }
  return Result;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SymbolLocation &S) {
  switch (S.kind()) {
  case SymbolLocation::Physical:
    // We can't decode the Location without SourceManager. Its raw
    // representation isn't completely useless (and distinguishes
    // SymbolReference from Symbol).
    return OS << "@0x"
              << llvm::utohexstr(
                     S.physical().getRawEncoding(), /*LowerCase=*/false,
                     /*Width=*/CHAR_BIT * sizeof(SourceLocation::UIntTy));
  case SymbolLocation::Standard:
    return OS << S.standard().scope() << S.standard().name();
  }
  llvm_unreachable("Unhandled Symbol kind");
}

bool Header::operator<(const Header &RHS) const {
  if (kind() != RHS.kind())
    return kind() < RHS.kind();
  switch (kind()) {
  case Header::Physical:
    return physical().getName() < RHS.physical().getName();
  case Header::Standard:
    return standard().name() < RHS.standard().name();
  case Header::Verbatim:
    return verbatim() < RHS.verbatim();
  }
  llvm_unreachable("unhandled Header kind");
}
} // namespace clang::include_cleaner
