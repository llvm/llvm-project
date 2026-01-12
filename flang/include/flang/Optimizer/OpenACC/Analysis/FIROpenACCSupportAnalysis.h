//===- FIROpenACCSupportAnalysis.h - FIR OpenACCSupport Analysis ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FIR-specific implementation of OpenACCSupport analysis.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_OPENACC_ANALYSIS_FIROPENACCSUPPORTANALYSIS_H
#define FORTRAN_OPTIMIZER_OPENACC_ANALYSIS_FIROPENACCSUPPORTANALYSIS_H

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Value.h"
#include <string>

namespace fir {
namespace acc {

/// FIR-specific implementation for the OpenACCSupport analysis interface.
///
/// This class provides the custom implementations of the OpenACCSupport
/// interface methods that are tailored to FIR's requirements and
/// can handle FIR dialect operations and types.
/// Its primary intent is to be registered with the OpenACCSupport analysis
/// using setImplementation()
///
/// Usage:
///   auto &support = getAnalysis<mlir::acc::OpenACCSupport>();
///   support.setImplementation(fir::acc::FIROpenACCSupportAnalysis());
///
class FIROpenACCSupportAnalysis {
public:
  FIROpenACCSupportAnalysis() = default;

  std::string getVariableName(mlir::Value v);

  std::string getRecipeName(mlir::acc::RecipeKind kind, mlir::Type type,
                            mlir::Value var);

  mlir::InFlightDiagnostic emitNYI(mlir::Location loc,
                                   const mlir::Twine &message);
};

} // namespace acc
} // namespace fir

#endif // FORTRAN_OPTIMIZER_OPENACC_ANALYSIS_FIROPENACCSUPPORTANALYSIS_H
