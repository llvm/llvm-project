//===- bolt/Utils/Utils.h - Common helper functions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common helper functions.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_UTILS_UTILS_H
#define BOLT_UTILS_UTILS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

namespace llvm {
class MCCFIInstruction;
namespace bolt {

/// Free memory allocated for \p List.
template <typename T> void clearList(T &List) {
  T TempList;
  TempList.swap(List);
}

void report_error(StringRef Message, std::error_code EC);

void report_error(StringRef Message, Error E);

void check_error(std::error_code EC, StringRef Message);

void check_error(Error E, Twine Message);

/// Return the name with escaped whitespace and backslash characters
std::string getEscapedName(const StringRef &Name);

/// Return the unescaped name
std::string getUnescapedName(const StringRef &Name);

// Determines which register a given DWARF expression is being assigned to.
// If the expression is defining the CFA, return NoneType.
Optional<uint8_t> readDWARFExpressionTargetReg(StringRef ExprBytes);

} // namespace bolt

bool operator==(const llvm::MCCFIInstruction &L,
                const llvm::MCCFIInstruction &R);

} // namespace llvm

#endif
