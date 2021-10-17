//===-- FirBuilder.h -- FIR operation builder -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder routines for constructing the FIR dialect of MLIR. As FIR is a
// dialect of MLIR, it makes extensive use of MLIR interfaces and MLIR's coding
// style (https://mlir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_FIRBUILDER_H
#define FORTRAN_OPTIMIZER_BUILDER_FIRBUILDER_H

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace fir {

//===----------------------------------------------------------------------===//
// FirOpBuilder
//===----------------------------------------------------------------------===//

/// Extends the MLIR OpBuilder to provide methods for building common FIR
/// patterns.
class FirOpBuilder : public mlir::OpBuilder {
public:
  explicit FirOpBuilder(mlir::Operation *op, const fir::KindMapping &kindMap)
      : OpBuilder{op} {}
  explicit FirOpBuilder(mlir::OpBuilder &builder,
                        const fir::KindMapping &kindMap)
      : OpBuilder{builder} {}

  /// Get the integer type whose bit width corresponds to the width of pointer
  /// types, or is bigger.
  mlir::Type getIntPtrType() {
    // TODO: Delay the need of such type until codegen or find a way to use
    // llvm::DataLayout::getPointerSizeInBits here.
    return getI64Type();
  }

  /// Create an integer constant of type \p type and value \p i.
  mlir::Value createIntegerConstant(mlir::Location loc, mlir::Type integerType,
                                    std::int64_t i);

  /// Lazy creation of fir.convert op.
  mlir::Value createConvert(mlir::Location loc, mlir::Type toTy,
                            mlir::Value val);

  /// Cast the input value to IndexType.
  mlir::Value convertToIndexType(mlir::Location loc, mlir::Value val) {
    return createConvert(loc, getIndexType(), val);
  }

  //===--------------------------------------------------------------------===//
  // If-Then-Else generation helper
  //===--------------------------------------------------------------------===//

  /// Helper class to create if-then-else in a structured way:
  /// Usage: genIfOp().genThen([&](){...}).genElse([&](){...}).end();
  /// Alternatively, getResults() can be used instead of end() to end the ifOp
  /// and get the ifOp results.
  class IfBuilder {
  public:
    IfBuilder(fir::IfOp ifOp, FirOpBuilder &builder)
        : ifOp{ifOp}, builder{builder} {}
    template <typename CC>
    IfBuilder &genThen(CC func) {
      builder.setInsertionPointToStart(&ifOp.thenRegion().front());
      func();
      return *this;
    }
    template <typename CC>
    IfBuilder &genElse(CC func) {
      assert(!ifOp.elseRegion().empty() && "must have else region");
      builder.setInsertionPointToStart(&ifOp.elseRegion().front());
      func();
      return *this;
    }
    void end() { builder.setInsertionPointAfter(ifOp); }

    /// End the IfOp and return the results if any.
    mlir::Operation::result_range getResults() {
      end();
      return ifOp.getResults();
    }

    fir::IfOp &getIfOp() { return ifOp; };

  private:
    fir::IfOp ifOp;
    FirOpBuilder &builder;
  };

  /// Create an IfOp and returns an IfBuilder that can generate the else/then
  /// bodies.
  IfBuilder genIfOp(mlir::Location loc, mlir::TypeRange results,
                    mlir::Value cdt, bool withElseRegion) {
    auto op = create<fir::IfOp>(loc, results, cdt, withElseRegion);
    return IfBuilder(op, *this);
  }

  /// Create an IfOp with no "else" region, and no result values.
  /// Usage: genIfThen(loc, cdt).genThen(lambda).end();
  IfBuilder genIfThen(mlir::Location loc, mlir::Value cdt) {
    auto op = create<fir::IfOp>(loc, llvm::None, cdt, false);
    return IfBuilder(op, *this);
  }

  /// Create an IfOp with an "else" region, and no result values.
  /// Usage: genIfThenElse(loc, cdt).genThen(lambda).genElse(lambda).end();
  IfBuilder genIfThenElse(mlir::Location loc, mlir::Value cdt) {
    auto op = create<fir::IfOp>(loc, llvm::None, cdt, true);
    return IfBuilder(op, *this);
  }

  /// Generate code testing \p addr is not a null address.
  mlir::Value genIsNotNull(mlir::Location loc, mlir::Value addr);

  /// Generate code testing \p addr is a null address.
  mlir::Value genIsNull(mlir::Location loc, mlir::Value addr);
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_BUILDER_FIRBUILDER_H
