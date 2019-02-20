//===-- IOStreamMacros.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_IOStreamMacros_h_
#define liblldb_IOStreamMacros_h_
#if defined(__cplusplus)

#include <iomanip>

#define RAW_HEXBASE std::setfill('0') << std::hex << std::right
#define HEXBASE '0' << 'x' << RAW_HEXBASE
#define RAWHEX8(x) RAW_HEXBASE << std::setw(2) << ((uint32_t)(x))
#define RAWHEX16 RAW_HEXBASE << std::setw(4)
#define RAWHEX32 RAW_HEXBASE << std::setw(8)
#define RAWHEX64 RAW_HEXBASE << std::setw(16)
#define HEX8(x) HEXBASE << std::setw(2) << ((uint32_t)(x))
#define HEX16 HEXBASE << std::setw(4)
#define HEX32 HEXBASE << std::setw(8)
#define HEX64 HEXBASE << std::setw(16)
#define RAW_HEX(x) RAW_HEXBASE << std::setw(sizeof(x) * 2) << (x)
#define HEX(x) HEXBASE << std::setw(sizeof(x) * 2) << (x)
#define HEX_SIZE(x, sz) HEXBASE << std::setw((sz)) << (x)
#define STRING_WIDTH(w) std::setfill(' ') << std::setw(w)
#define LEFT_STRING_WIDTH(s, w)                                                \
  std::left << std::setfill(' ') << std::setw(w) << (s) << std::right
#define DECIMAL std::dec << std::setfill(' ')
#define DECIMAL_WIDTH(w) DECIMAL << std::setw(w)
//#define FLOAT(n, d)       std::setfill(' ') << std::setw((n)+(d)+1) <<
//std::setprecision(d) << std::showpoint << std::fixed
#define INDENT_WITH_SPACES(iword_idx)                                          \
  std::setfill(' ') << std::setw((iword_idx)) << ""
#define INDENT_WITH_TABS(iword_idx)                                            \
  std::setfill('\t') << std::setw((iword_idx)) << ""

#endif // #if defined(__cplusplus)
#endif // liblldb_IOStreamMacros_h_
