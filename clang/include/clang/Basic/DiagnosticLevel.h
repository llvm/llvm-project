//===--- DiagnosticLevel.h - Diagnostic Severity Level-----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the DiagnosticLevel enum.
///
/// This file has no other dependencies on Clang headers, to ensure that it can
/// be included from tblgen.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_DIAGNOSTICLEVEL_H
#define LLVM_CLANG_BASIC_DIAGNOSTICLEVEL_H

#include <cstdint>

namespace clang {

/// The level of the diagnostic, after it has been through mapping.
enum class DiagnosticLevel : uint8_t {
  Ignored,
  Note,
  Remark,
  Warning,
  Error,
  Fatal
};

} // namespace clang

#endif
