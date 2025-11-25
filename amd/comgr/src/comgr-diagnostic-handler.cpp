//===- comgr-diagnostic-handler.cpp - Handle LLVM diagnostics -------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the handling of LLVM diagnonstics, which are generated
/// during LLVM API interactions. We forward these to the Comgr Log to aid in
/// debugging.
///
//===----------------------------------------------------------------------===//

#include "comgr-diagnostic-handler.h"

#include "llvm/IR/DiagnosticPrinter.h"

namespace COMGR {
using namespace llvm;
bool AMDGPUCompilerDiagnosticHandler::handleDiagnostics(
    const DiagnosticInfo &DI) {
  unsigned Severity = DI.getSeverity();
  switch (Severity) {
  case DS_Error:
    LogS << "ERROR: ";
    break;
  case DS_Warning:
    LogS << "WARNING: ";
    break;
  case DS_Remark:
    LogS << "REMARK: ";
    break;
  case DS_Note:
    LogS << "NOTE: ";
    break;
  default:
    LogS << "(Unknown DiagnosticInfo Severity): ";
    break;
  }
  DiagnosticPrinterRawOStream DP(LogS);
  DI.print(DP);
  LogS << "\n";
  return true;
}
} // namespace COMGR
