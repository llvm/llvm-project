//===-- HlfirIntrinsics.h -- lowering to HLFIR intrinsic ops ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//
///
/// Implements lowering of transformational intrinsics to HLFIR intrinsic
/// operations
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_HLFIRINTRINSICS_H
#define FORTRAN_LOWER_HLFIRINTRINSICS_H

#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <optional>
#include <string>

namespace mlir {
class Location;
class Type;
class Value;
class ValueRange;
} // namespace mlir

namespace fir {
class FirOpBuilder;
struct IntrinsicArgumentLoweringRules;
} // namespace fir

namespace Fortran::lower {

/// This structure holds the initial lowered value of an actual argument that
/// was lowered regardless of the interface, and it holds whether or not it
/// may be absent at runtime and the dummy is optional.
struct PreparedActualArgument {

  PreparedActualArgument(hlfir::Entity actual,
                         std::optional<mlir::Value> isPresent)
      : actual{actual}, isPresent{isPresent} {}
  void setElementalIndices(mlir::ValueRange &indices) {
    oneBasedElementalIndices = &indices;
  }
  hlfir::Entity getActual(mlir::Location loc,
                          fir::FirOpBuilder &builder) const {
    if (oneBasedElementalIndices)
      return hlfir::getElementAt(loc, builder, actual,
                                 *oneBasedElementalIndices);
    return actual;
  }
  hlfir::Entity getOriginalActual() const { return actual; }
  void setOriginalActual(hlfir::Entity newActual) { actual = newActual; }
  bool handleDynamicOptional() const { return isPresent.has_value(); }
  mlir::Value getIsPresent() const {
    assert(handleDynamicOptional() && "not a dynamic optional");
    return *isPresent;
  }

  void resetOptionalAspect() { isPresent = std::nullopt; }

private:
  hlfir::Entity actual;
  mlir::ValueRange *oneBasedElementalIndices{nullptr};
  // When the actual may be dynamically optional, "isPresent"
  // holds a boolean value indicating the presence of the
  // actual argument at runtime.
  std::optional<mlir::Value> isPresent;
};

/// Vector of pre-lowered actual arguments. nullopt if the actual is
/// "statically" absent (if it was not syntactically  provided).
using PreparedActualArguments =
    llvm::SmallVector<std::optional<PreparedActualArgument>>;

std::optional<hlfir::EntityWithAttributes> lowerHlfirIntrinsic(
    fir::FirOpBuilder &builder, mlir::Location loc, const std::string &name,
    const Fortran::lower::PreparedActualArguments &loweredActuals,
    const fir::IntrinsicArgumentLoweringRules *argLowering,
    mlir::Type stmtResultType);

} // namespace Fortran::lower
#endif // FORTRAN_LOWER_HLFIRINTRINSICS_H
