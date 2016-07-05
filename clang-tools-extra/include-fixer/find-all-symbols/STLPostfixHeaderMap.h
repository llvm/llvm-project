//===-- STLPostfixHeaderMap.h - hardcoded header map for STL ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
