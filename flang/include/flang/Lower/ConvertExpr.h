//===-- Lower/ConvertExpr.h -- lowering of expressions ----------*- C++ -*-===//
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
/// Implements the conversion from Fortran::evaluate::Expr trees to FIR.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CONVERTEXPR_H
#define FORTRAN_LOWER_CONVERTEXPR_H

#include "flang/Lower/Support/BoxValue.h"

namespace mlir {
class Location;
class Value;
} // namespace mlir

namespace fir {
class ShapeOp;
}

namespace Fortran {
namespace evaluate {
template <typename>
class Expr;
struct SomeType;
} // namespace evaluate

namespace lower {

class AbstractConverter;
class SymMap;

/// The evaluation of some expressions implies a surrounding context. This
/// context is abstracted by this class.
class ExpressionContext {
public:
  ExpressionContext() = default;

  //===--------------------------------------------------------------------===//
  // Expression is in an array context.
  //===--------------------------------------------------------------------===//

  ExpressionContext(mlir::Value shape) : shape{shape}, preludePhase{true} {}

  bool inArrayContext() const { return shape ? true : false; }
  bool inPreludePhase() const { return preludePhase; }
  fir::ShapeOp getShape() const;
  llvm::ArrayRef<mlir::Value> getLoopCounters() const { return loopCounters; }
  llvm::ArrayRef<mlir::Value> getArrayBlockArgs() const {
    return arrayBlockArgs;
  }
  llvm::ArrayRef<mlir::Value> getLoopReturnVals() const {
    return loopReturnVals;
  }

  ExpressionContext &setLoopPhase(bool inPrelude = false) {
    preludePhase = inPrelude;
    return *this;
  }

  void addLoop(mlir::Value counter, mlir::Value blockArg, mlir::Value result);
  void finalizeLoopNest();

  //===--------------------------------------------------------------------===//
  // Expression is in an initializer context.
  //===--------------------------------------------------------------------===//

  bool inInitializer() const { return isInitializer; }
  ExpressionContext &setInInitializer(bool val = true) {
    isInitializer = val;
    return *this;
  }

private:
  mlir::Value shape; // shape op
  std::vector<mlir::Value> loopCounters;
  std::vector<mlir::Value> arrayBlockArgs;
  std::vector<mlir::Value> loopReturnVals;
  bool preludePhase{false};

  bool isInitializer{false};
};

/// Create an extended expression value.
fir::ExtendedValue
createSomeExtendedExpression(mlir::Location loc, AbstractConverter &converter,
                             const evaluate::Expr<evaluate::SomeType> &expr,
                             SymMap &symMap, const ExpressionContext &context);

/// Create an extended expression address.
fir::ExtendedValue
createSomeExtendedAddress(mlir::Location loc, AbstractConverter &converter,
                          const evaluate::Expr<evaluate::SomeType> &expr,
                          SymMap &symMap, const ExpressionContext &context);

/// Create a string literal. Lowers `str` to the MLIR representation of a
/// literal CHARACTER value. (KIND is assumed to be 1.)
fir::ExtendedValue createStringLiteral(mlir::Location loc,
                                       AbstractConverter &converter,
                                       llvm::StringRef str, std::uint64_t len);

/// Create a shape op from an extended value, exv.
/// TODO: Do we want to keep a shape op in the extended value?
mlir::Value createShape(mlir::Location loc, AbstractConverter &converter,
                        const fir::ExtendedValue &exv);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_CONVERTEXPR_H
