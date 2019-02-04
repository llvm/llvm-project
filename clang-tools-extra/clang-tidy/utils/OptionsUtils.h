//===--- DanglingHandleCheck.h - clang-tidy----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_OPTIONUTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_OPTIONUTILS_H

#include "../ClangTidy.h"

namespace clang {
namespace tidy {
namespace utils {
namespace options {

/// \brief Parse a semicolon separated list of strings.
std::vector<std::string> parseStringList(StringRef Option);

/// \brief Serialize a sequence of names that can be parsed by
/// ``parseStringList``.
std::string serializeStringList(ArrayRef<std::string> Strings);

} // namespace options
} // namespace utils
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_OPTIONUTILS_H
