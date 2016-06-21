//===--- DanglingHandleCheck.h - clang-tidy----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
