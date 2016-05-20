//===--- DanglingHandleCheck.cpp - clang-tidy------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
