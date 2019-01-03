//===--- SymbolName.cpp - Clang refactoring library -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactor/SymbolName.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace tooling {

static void initNames(std::vector<std::string> &Strings, StringRef Name,
                      bool IsObjectiveCSelector) {
  if (!IsObjectiveCSelector) {
    Strings.push_back(Name.str());
    return;
  }
  // Decompose an Objective-C selector name into multiple strings.
  do {
    auto StringAndName = Name.split(':');
    Strings.push_back(StringAndName.first.str());
    Name = StringAndName.second;
  } while (!Name.empty());
}

OldSymbolName::OldSymbolName(StringRef Name, const LangOptions &LangOpts) {
  initNames(Strings, Name, LangOpts.ObjC);
}

OldSymbolName::OldSymbolName(StringRef Name, bool IsObjectiveCSelector) {
  initNames(Strings, Name, IsObjectiveCSelector);
}

OldSymbolName::OldSymbolName(ArrayRef<StringRef> Name) {
  for (const auto &Piece : Name)
    Strings.push_back(Piece.str());
}

void OldSymbolName::print(raw_ostream &OS) const {
  for (size_t I = 0, E = Strings.size(); I != E; ++I) {
    if (I != 0)
      OS << ':';
    OS << Strings[I];
  }
}

raw_ostream &operator<<(raw_ostream &OS, const OldSymbolName &N) {
  N.print(OS);
  return OS;
}

} // end namespace tooling
} // end namespace clang
