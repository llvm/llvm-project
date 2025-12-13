//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OptionsUtils.h"
#include "llvm/ADT/StringExtras.h"

namespace clang::tidy::utils::options {

static const char StringsDelimiter[] = ";";

std::vector<StringRef> parseStringList(StringRef Option) {
  Option = Option.trim().trim(StringsDelimiter);
  if (Option.empty())
    return {};
  std::vector<StringRef> Result;
  Result.reserve(Option.count(StringsDelimiter) + 1);
  StringRef Cur;
  while (std::tie(Cur, Option) = Option.split(StringsDelimiter),
         !Option.empty()) {
    Cur = Cur.trim();
    if (!Cur.empty())
      Result.push_back(Cur);
  }
  Cur = Cur.trim();
  if (!Cur.empty())
    Result.push_back(Cur);
  return Result;
}

std::vector<StringRef> parseListPair(StringRef L, StringRef R) {
  L = L.trim().trim(StringsDelimiter);
  if (L.empty())
    return parseStringList(R);
  R = R.trim().trim(StringsDelimiter);
  if (R.empty())
    return parseStringList(L);
  std::vector<StringRef> Result;
  Result.reserve(2 + L.count(StringsDelimiter) + R.count(StringsDelimiter));
  for (StringRef Option : {L, R}) {
    StringRef Cur;
    while (std::tie(Cur, Option) = Option.split(StringsDelimiter),
           !Option.empty()) {
      Cur = Cur.trim();
      if (!Cur.empty())
        Result.push_back(Cur);
    }
    Cur = Cur.trim();
    if (!Cur.empty())
      Result.push_back(Cur);
  }
  return Result;
}

std::string serializeStringList(ArrayRef<StringRef> Strings) {
  return llvm::join(Strings, StringsDelimiter);
}

} // namespace clang::tidy::utils::options
