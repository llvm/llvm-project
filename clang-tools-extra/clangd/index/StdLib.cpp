//===-- StdLib.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "StdLib.h"
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Compiler.h"
#include "Config.h"
#include "SymbolCollector.h"
#include "index/IndexAction.h"
#include "support/Logger.h"
#include "support/ThreadsafeFS.h"
#include "support/Trace.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace clangd {
namespace {

enum Lang { C, CXX };

Lang langFromOpts(const LangOptions &LO) { return LO.CPlusPlus ? CXX : C; }
llvm::StringLiteral mandatoryHeader(Lang L) {
  switch (L) {
  case C:
    return "stdio.h";
  case CXX:
    return "vector";
  }
  llvm_unreachable("unhandled Lang");
}

LangStandard::Kind standardFromOpts(const LangOptions &LO) {
  if (LO.CPlusPlus) {
    if (LO.CPlusPlus23)
      return LangStandard::lang_cxx23;
    if (LO.CPlusPlus20)
      return LangStandard::lang_cxx20;
    if (LO.CPlusPlus17)
      return LangStandard::lang_cxx17;
    if (LO.CPlusPlus14)
      return LangStandard::lang_cxx14;
    if (LO.CPlusPlus11)
      return LangStandard::lang_cxx11;
    return LangStandard::lang_cxx98;
  }
  if (LO.C23)
    return LangStandard::lang_c23;
  // C17 has no new features, so treat {C11,C17} as C17.
  if (LO.C11)
    return LangStandard::lang_c17;
  return LangStandard::lang_c99;
}

std::string buildUmbrella(llvm::StringLiteral Mandatory,
                          llvm::ArrayRef<tooling::stdlib::Header> Headers) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);

  // We __has_include guard all our #includes to avoid errors when using older
  // stdlib version that don't have headers for the newest language standards.
  // But make sure we get *some* error if things are totally broken.
  OS << llvm::formatv(
      "#if !__has_include(<{0}>)\n"
      "#error Mandatory header <{0}> not found in standard library!\n"
      "#endif\n",
      Mandatory);

  for (auto Header : Headers) {
    OS << llvm::formatv("#if __has_include({0})\n"
                        "#include {0}\n"
                        "#endif\n",
                        Header);
  }
  return Result;
}

} // namespace

llvm::StringRef getStdlibUmbrellaHeader(const LangOptions &LO) {
  // The umbrella header is the same for all versions of each language.
  // Headers that are unsupported in old lang versions are usually guarded by
  // #if. Some headers may be not present in old stdlib versions, the umbrella
  // header guards with __has_include for this purpose.
  Lang L = langFromOpts(LO);
  switch (L) {
  case CXX:
    static std::string *UmbrellaCXX = new std::string(buildUmbrella(
        mandatoryHeader(L),
        tooling::stdlib::Header::all(tooling::stdlib::Lang::CXX)));
    return *UmbrellaCXX;
  case C:
    static std::string *UmbrellaC = new std::string(
        buildUmbrella(mandatoryHeader(L),
                      tooling::stdlib::Header::all(tooling::stdlib::Lang::C)));
    return *UmbrellaC;
  }
  llvm_unreachable("invalid Lang in langFromOpts");
}

namespace {

// Including the standard library leaks unwanted transitively included symbols.
//
// We want to drop these, they're a bit tricky to identify:
//  - we don't want to limit to symbols on our list, as our list has only
//    top-level symbols (and there may be legitimate stdlib extensions).
//  - we can't limit to only symbols defined in known stdlib headers, as stdlib
//    internal structure is murky
//  - we can't strictly require symbols to come from a particular path, e.g.
//      libstdc++ is mostly under /usr/include/c++/10/...
//      but std::ctype_base is under /usr/include/<platform>/c++/10/...
// We require the symbol to come from a header that is *either* from
// the standard library path (as identified by the location of <vector>), or
// another header that defines a symbol from our stdlib list.
SymbolSlab filter(SymbolSlab Slab, const StdLibLocation &Loc) {
  SymbolSlab::Builder Result;

  static auto &StandardHeaders = *[] {
    auto *Set = new llvm::DenseSet<llvm::StringRef>();
    for (auto Header : tooling::stdlib::Header::all(tooling::stdlib::Lang::CXX))
      Set->insert(Header.name());
    for (auto Header : tooling::stdlib::Header::all(tooling::stdlib::Lang::C))
      Set->insert(Header.name());
    return Set;
  }();

  // Form prefixes like file:///usr/include/c++/10/
  // These can be trivially prefix-compared with URIs in the indexed symbols.
  llvm::SmallVector<std::string> StdLibURIPrefixes;
  for (const auto &Path : Loc.Paths) {
    StdLibURIPrefixes.push_back(URI::create(Path).toString());
    if (StdLibURIPrefixes.back().back() != '/')
      StdLibURIPrefixes.back().push_back('/');
  }
  // For each header URI, is it *either* prefixed by StdLibURIPrefixes *or*
  // owner of a symbol whose insertable header is in StandardHeaders?
  // Pointer key because strings in a SymbolSlab are interned.
  llvm::DenseMap<const char *, bool> GoodHeader;
  for (const Symbol &S : Slab) {
    if (!S.IncludeHeaders.empty() &&
        StandardHeaders.contains(S.IncludeHeaders.front().IncludeHeader)) {
      GoodHeader[S.CanonicalDeclaration.FileURI] = true;
      GoodHeader[S.Definition.FileURI] = true;
      continue;
    }
    for (const char *URI :
         {S.CanonicalDeclaration.FileURI, S.Definition.FileURI}) {
      auto R = GoodHeader.try_emplace(URI, false);
      if (R.second) {
        R.first->second = llvm::any_of(
            StdLibURIPrefixes,
            [&, URIStr(llvm::StringRef(URI))](const std::string &Prefix) {
              return URIStr.starts_with(Prefix);
            });
      }
    }
  }
#ifndef NDEBUG
  for (const auto &Good : GoodHeader)
    if (Good.second && *Good.first)
      dlog("Stdlib header: {0}", Good.first);
#endif
  // Empty URIs aren't considered good. (Definition can be blank).
  auto IsGoodHeader = [&](const char *C) { return *C && GoodHeader.lookup(C); };

  for (const Symbol &S : Slab) {
    if (!(IsGoodHeader(S.CanonicalDeclaration.FileURI) ||
          IsGoodHeader(S.Definition.FileURI))) {
      dlog("Ignoring wrong-header symbol {0}{1} in {2}", S.Scope, S.Name,
           S.CanonicalDeclaration.FileURI);
      continue;
    }
    Result.insert(S);
  }

  return std::move(Result).build();
}

} // namespace

SymbolSlab indexStandardLibrary(llvm::StringRef HeaderSources,
                                std::unique_ptr<CompilerInvocation> CI,
                                const StdLibLocation &Loc,
                                const ThreadsafeFS &TFS) {
  if (CI->getFrontendOpts().Inputs.size() != 1 ||
      !CI->getPreprocessorOpts().ImplicitPCHInclude.empty()) {
    elog("Indexing standard library failed: bad CompilerInvocation");
    assert(false && "indexing stdlib with a dubious CompilerInvocation!");
    return SymbolSlab();
  }
  const FrontendInputFile &Input = CI->getFrontendOpts().Inputs.front();
  trace::Span Tracer("StandardLibraryIndex");
  LangStandard::Kind LangStd = standardFromOpts(CI->getLangOpts());
  log("Indexing {0} standard library in the context of {1}",
      LangStandard::getLangStandardForKind(LangStd).getName(), Input.getFile());

  SymbolSlab Symbols;
  IgnoreDiagnostics IgnoreDiags;
  // CompilerInvocation is taken from elsewhere, and may map a dirty buffer.
  CI->getPreprocessorOpts().clearRemappedFiles();
  auto Clang = prepareCompilerInstance(
      std::move(CI), /*Preamble=*/nullptr,
      llvm::MemoryBuffer::getMemBuffer(HeaderSources, Input.getFile()),
      TFS.view(/*CWD=*/std::nullopt), IgnoreDiags);
  if (!Clang) {
    elog("Standard Library Index: Couldn't build compiler instance");
    return Symbols;
  }

  SymbolCollector::Options IndexOpts;
  IndexOpts.Origin = SymbolOrigin::StdLib;
  IndexOpts.CollectMainFileSymbols = false;
  IndexOpts.CollectMainFileRefs = false;
  IndexOpts.CollectMacro = true;
  IndexOpts.StoreAllDocumentation = true;
  // Sadly we can't use IndexOpts.FileFilter to restrict indexing scope.
  // Files from outside the StdLibLocation may define true std symbols anyway.
  // We end up "blessing" such headers, and can only do that by indexing
  // everything first.

  // Refs, relations, include graph in the stdlib mostly aren't useful.
  auto Action = createStaticIndexingAction(
      IndexOpts, [&](SymbolSlab S) { Symbols = std::move(S); }, nullptr,
      nullptr, nullptr);

  if (!Action->BeginSourceFile(*Clang, Input)) {
    elog("Standard Library Index: BeginSourceFile() failed");
    return Symbols;
  }

  if (llvm::Error Err = Action->Execute()) {
    elog("Standard Library Index: Execute failed: {0}", std::move(Err));
    return Symbols;
  }

  Action->EndSourceFile();

  unsigned SymbolsBeforeFilter = Symbols.size();
  Symbols = filter(std::move(Symbols), Loc);
  bool Errors = Clang->hasDiagnostics() &&
                Clang->getDiagnostics().hasUncompilableErrorOccurred();
  log("Indexed {0} standard library{3}: {1} symbols, {2} filtered",
      LangStandard::getLangStandardForKind(LangStd).getName(), Symbols.size(),
      SymbolsBeforeFilter - Symbols.size(),
      Errors ? " (incomplete due to errors)" : "");
  SPAN_ATTACH(Tracer, "symbols", int(Symbols.size()));
  return Symbols;
}

SymbolSlab indexStandardLibrary(std::unique_ptr<CompilerInvocation> Invocation,
                                const StdLibLocation &Loc,
                                const ThreadsafeFS &TFS) {
  llvm::StringRef Header = getStdlibUmbrellaHeader(Invocation->getLangOpts());
  return indexStandardLibrary(Header, std::move(Invocation), Loc, TFS);
}

bool StdLibSet::isBest(const LangOptions &LO) const {
  return standardFromOpts(LO) >=
         Best[langFromOpts(LO)].load(std::memory_order_acquire);
}

std::optional<StdLibLocation> StdLibSet::add(const LangOptions &LO,
                                             const HeaderSearch &HS) {
  Lang L = langFromOpts(LO);
  int OldVersion = Best[L].load(std::memory_order_acquire);
  int NewVersion = standardFromOpts(LO);
  dlog("Index stdlib? {0}",
       LangStandard::getLangStandardForKind(standardFromOpts(LO)).getName());

  if (!Config::current().Index.StandardLibrary) {
    dlog("No: disabled in config");
    return std::nullopt;
  }

  if (NewVersion <= OldVersion) {
    dlog("No: have {0}, {1}>={2}",
         LangStandard::getLangStandardForKind(
             static_cast<LangStandard::Kind>(NewVersion))
             .getName(),
         OldVersion, NewVersion);
    return std::nullopt;
  }

  // We'd like to index a standard library here if there is one.
  // Check for the existence of <vector> on the search path.
  // We could cache this, but we only get here repeatedly when there's no
  // stdlib, and even then only once per preamble build.
  llvm::StringLiteral ProbeHeader = mandatoryHeader(L);
  llvm::SmallString<256> Path; // Scratch space.
  llvm::SmallVector<std::string> SearchPaths;
  auto RecordHeaderPath = [&](llvm::StringRef HeaderPath) {
    llvm::StringRef DirPath = llvm::sys::path::parent_path(HeaderPath);
    if (!HS.getFileMgr().getVirtualFileSystem().getRealPath(DirPath, Path))
      SearchPaths.emplace_back(Path);
  };
  for (const auto &DL :
       llvm::make_range(HS.search_dir_begin(), HS.search_dir_end())) {
    switch (DL.getLookupType()) {
    case DirectoryLookup::LT_NormalDir: {
      Path = DL.getDirRef()->getName();
      llvm::sys::path::append(Path, ProbeHeader);
      llvm::vfs::Status Stat;
      if (!HS.getFileMgr().getNoncachedStatValue(Path, Stat) &&
          Stat.isRegularFile())
        RecordHeaderPath(Path);
      break;
    }
    case DirectoryLookup::LT_Framework:
      // stdlib can't be a framework (framework includes must have a slash)
      continue;
    case DirectoryLookup::LT_HeaderMap:
      llvm::StringRef Target =
          DL.getHeaderMap()->lookupFilename(ProbeHeader, Path);
      if (!Target.empty())
        RecordHeaderPath(Target);
      break;
    }
  }
  if (SearchPaths.empty())
    return std::nullopt;

  dlog("Found standard library in {0}", llvm::join(SearchPaths, ", "));

  while (!Best[L].compare_exchange_weak(OldVersion, NewVersion,
                                        std::memory_order_acq_rel))
    if (OldVersion >= NewVersion) {
      dlog("No: lost the race");
      return std::nullopt; // Another thread won the race while we were
                           // checking.
    }

  dlog("Yes, index stdlib!");
  return StdLibLocation{std::move(SearchPaths)};
}

} // namespace clangd
} // namespace clang
