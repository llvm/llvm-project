//===-- OptionsUtils.cpp - clang-tidy -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OptionsUtils.h"

namespace clang {
namespace tidy {
namespace utils {
namespace options {

static const char StringsDelimiter[] = ";";

std::vector<std::string> parseStringList(StringRef Option) {
  SmallVector<StringRef, 4> Names;
  Option.split(Names, StringsDelimiter);
  std::vector<std::string> Result;
  for (StringRef &Name : Names) {
    Name = Name.trim();
    if (!Name.empty())
      Result.push_back(Name);
  }
  return Result;
}

std::string serializeStringList(ArrayRef<std::string> Strings) {
  return llvm::join(Strings.begin(), Strings.end(), StringsDelimiter);
}

} // namespace options
} // namespace utils
} // namespace tidy
} // namespace clang
