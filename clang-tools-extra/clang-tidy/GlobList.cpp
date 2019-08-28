//===--- tools/extra/clang-tidy/GlobList.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GlobList.h"
#include "llvm/ADT/SmallString.h"

using namespace clang;
using namespace tidy;

// Returns true if GlobList starts with the negative indicator ('-'), removes it
// from the GlobList.
static bool ConsumeNegativeIndicator(StringRef &GlobList) {
  GlobList = GlobList.trim(" \r\n");
  if (GlobList.startswith("-")) {
    GlobList = GlobList.substr(1);
    return true;
  }
  return false;
}

// Converts first glob from the comma-separated list of globs to Regex and
// removes it and the trailing comma from the GlobList.
static llvm::Regex ConsumeGlob(StringRef &GlobList) {
  StringRef UntrimmedGlob = GlobList.substr(0, GlobList.find(','));
  StringRef Glob = UntrimmedGlob.trim(' ');
  GlobList = GlobList.substr(UntrimmedGlob.size() + 1);
  SmallString<128> RegexText("^");
  StringRef MetaChars("()^$|*+?.[]\\{}");
  for (char C : Glob) {
    if (C == '*')
      RegexText.push_back('.');
    else if (MetaChars.find(C) != StringRef::npos)
      RegexText.push_back('\\');
    RegexText.push_back(C);
  }
  RegexText.push_back('$');
  return llvm::Regex(RegexText);
}

GlobList::GlobList(StringRef Globs) {
  do {
    GlobListItem Item;
    Item.IsPositive = !ConsumeNegativeIndicator(Globs);
    Item.Regex = ConsumeGlob(Globs);
    Items.push_back(std::move(Item));
  } while (!Globs.empty());
}

bool GlobList::contains(StringRef S) {
  bool Contains = false;
  for (const GlobListItem &Item : Items) {
    if (Item.Regex.match(S))
      Contains = Item.IsPositive;
  }
  return Contains;
}
