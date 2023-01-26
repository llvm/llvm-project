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

/// LTO-generated function names take a form:
///
///   <function_name>.lto_priv.<decimal_number>/...
///     or
///   <function_name>.constprop.<decimal_number>/...
///
/// they can also be:
///
///   <function_name>.lto_priv.<decimal_number1>.lto_priv.<decimal_number2>/...
///
/// The <decimal_number> is a global counter used for the whole program. As a
/// result, a tiny change in a program may affect the naming of many LTO
/// functions. For us this means that if we do a precise name matching, then
/// a large set of functions could be left without a profile.
///
/// To solve this issue, we try to match a function to any profile:
///
///   <function_name>.(lto_priv|consprop).*
///
/// The name before an asterisk above represents a common LTO name for a family
/// of functions. Later, out of all matching profiles we pick the one with the
/// best match.
///
/// Return a common part of LTO name for a given \p Name.
std::optional<StringRef> getLTOCommonName(const StringRef Name);

// Determines which register a given DWARF expression is being assigned to.
// If the expression is defining the CFA, return std::nullopt.
std::optional<uint8_t> readDWARFExpressionTargetReg(StringRef ExprBytes);

} // namespace bolt

bool operator==(const llvm::MCCFIInstruction &L,
                const llvm::MCCFIInstruction &R);

} // namespace llvm

#endif
