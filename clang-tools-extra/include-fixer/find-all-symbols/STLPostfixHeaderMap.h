//===-- STLPostfixHeaderMap.h - hardcoded header map for STL ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_TOOL_STL_POSTFIX_HEADER_MAP_H
#define LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_TOOL_STL_POSTFIX_HEADER_MAP_H

#include "HeaderMapCollector.h"

namespace clang {
namespace find_all_symbols {

const HeaderMapCollector::RegexHeaderMap *getSTLPostfixHeaderMap();

} // namespace find_all_symbols
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_TOOL_STL_POSTFIX_HEADER_MAP_H
