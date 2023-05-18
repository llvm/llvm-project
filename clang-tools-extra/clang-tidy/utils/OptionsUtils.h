//===--- DanglingHandleCheck.h - clang-tidy----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_OPTIONUTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_OPTIONUTILS_H

#include "clang/Basic/LLVM.h"
#include <string>
#include <vector>

namespace clang::tidy::utils::options {

/// Parse a semicolon separated list of strings.
std::vector<StringRef> parseStringList(StringRef Option);

std::vector<StringRef> parseListPair(StringRef L, StringRef R);

/// Serialize a sequence of names that can be parsed by
/// ``parseStringList``.
std::string serializeStringList(ArrayRef<StringRef> Strings);

} // namespace clang::tidy::utils::options

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_OPTIONUTILS_H
