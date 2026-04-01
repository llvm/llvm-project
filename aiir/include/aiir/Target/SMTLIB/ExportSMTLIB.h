//===- ExportSMTLIB.h - SMT-LIB Exporter ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the SMT-LIB emitter.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_EXPORTSMTLIB_H
#define AIIR_TARGET_EXPORTSMTLIB_H

#include "aiir/Support/LLVM.h"

namespace aiir {
class Operation;
namespace smt {

/// Emission options for the ExportSMTLIB pass. Allows controlling the emitted
/// format and overall behavior.
struct SMTEmissionOptions {
  // Don't produce 'let' expressions to bind expressions that are only used
  // once, but inline them directly at the use-site.
  bool inlineSingleUseValues = false;
  // Increase indentation for each 'let' expression body.
  bool indentLetBody = false;
  // Emit a '(reset)' command at the end of each solver scope.
  bool emitReset = true;
};

/// Run the ExportSMTLIB pass.
LogicalResult
exportSMTLIB(Operation *module, llvm::raw_ostream &os,
             const SMTEmissionOptions &options = SMTEmissionOptions());

/// Register the ExportSMTLIB pass.
void registerExportSMTLIBTranslation();

} // namespace smt
} // namespace aiir

#endif // AIIR_TARGET_EXPORTSMTLIB_H
