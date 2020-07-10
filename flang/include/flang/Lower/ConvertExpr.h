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
#include <cstdint>

namespace mlir {
class Location;
class OpBuilder;
class Type;
class Value;
} // namespace mlir

namespace fir {
class AllocaExpr;
} // namespace fir

namespace Fortran {
namespace common {
class IntrinsicTypeDefaultKinds;
} // namespace common

namespace evaluate {
template <typename>
class Expr;
struct SomeType;
} // namespace evaluate

namespace semantics {
class Symbol;
} // namespace semantics

namespace lower {

class AbstractConverter;
class FirOpBuilder;
class SymMap;

/// The evaluation of some expressions implies a surrounding context. This
/// context is abstracted by this class.
class ExpressionContext {
public:
  ExpressionContext() = default;
  ExpressionContext(llvm::ArrayRef<mlir::Value> lcvs)
      : loopVars{lcvs.begin(), lcvs.end()} {}

  bool inArrayContext() const { return loopVars.size() > 0; }
  const std::vector<mlir::Value> &getLoopVars() const { return loopVars; }

private:
  std::vector<mlir::Value> loopVars{};
};

/// Create an expression.
/// Lowers `expr` to the FIR dialect of MLIR. The expression is lowered to a
/// value result.
mlir::Value createSomeExpression(mlir::Location loc,
                                 AbstractConverter &converter,
                                 const evaluate::Expr<evaluate::SomeType> &expr,
                                 SymMap &symMap);

/// Create an extended expression value.
fir::ExtendedValue
createSomeExtendedExpression(mlir::Location loc, AbstractConverter &converter,
                             const evaluate::Expr<evaluate::SomeType> &expr,
                             SymMap &symMap, const ExpressionContext &context);

/// Create an address.
/// Lowers `expr` to the FIR dialect of MLIR. The expression must be an entity
/// and the address of the entity is returned.
mlir::Value createSomeAddress(mlir::Location loc, AbstractConverter &converter,
                              const evaluate::Expr<evaluate::SomeType> &expr,
                              SymMap &symMap);

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

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_CONVERTEXPR_H
