//===-- Lower/OpenMP/ReductionProcessor.h -----------------------*- C++ -*-===//
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

#ifndef FORTRAN_LOWER_REDUCTIONPROCESSOR_H
#define FORTRAN_LOWER_REDUCTIONPROCESSOR_H

#include "Clauses.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/type.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace omp {
class DeclareReductionOp;
} // namespace omp
} // namespace mlir

namespace Fortran {
namespace lower {
class AbstractConverter;
} // namespace lower
} // namespace Fortran

namespace Fortran {
namespace lower {
namespace omp {

class ReductionProcessor {
public:
  // TODO: Move this enumeration to the OpenMP dialect
  enum ReductionIdentifier {
    ID,
    USER_DEF_OP,
    ADD,
    SUBTRACT,
    MULTIPLY,
    AND,
    OR,
    EQV,
    NEQV,
    MAX,
    MIN,
    IAND,
    IOR,
    IEOR
  };

  static ReductionIdentifier
  getReductionType(const omp::clause::ProcedureDesignator &pd);

  static ReductionIdentifier
  getReductionType(omp::clause::DefinedOperator::IntrinsicOperator intrinsicOp);

  static bool
  supportedIntrinsicProcReduction(const omp::clause::ProcedureDesignator &pd);

  static const Fortran::semantics::SourceName
  getRealName(const Fortran::semantics::Symbol *symbol);

  static const Fortran::semantics::SourceName
  getRealName(const omp::clause::ProcedureDesignator &pd);

  static bool
  doReductionByRef(const llvm::SmallVectorImpl<mlir::Value> &reductionVars);

  static std::string getReductionName(llvm::StringRef name,
                                      const fir::KindMapping &kindMap,
                                      mlir::Type ty, bool isByRef);

  static std::string
  getReductionName(omp::clause::DefinedOperator::IntrinsicOperator intrinsicOp,
                   const fir::KindMapping &kindMap, mlir::Type ty,
                   bool isByRef);

  /// This function returns the identity value of the operator \p
  /// reductionOpName. For example:
  ///    0 + x = x,
  ///    1 * x = x
  static int getOperationIdentity(ReductionIdentifier redId,
                                  mlir::Location loc);

  static mlir::Value getReductionInitValue(mlir::Location loc, mlir::Type type,
                                           ReductionIdentifier redId,
                                           fir::FirOpBuilder &builder);

  template <typename FloatOp, typename IntegerOp>
  static mlir::Value getReductionOperation(fir::FirOpBuilder &builder,
                                           mlir::Type type, mlir::Location loc,
                                           mlir::Value op1, mlir::Value op2);

  static mlir::Value createScalarCombiner(fir::FirOpBuilder &builder,
                                          mlir::Location loc,
                                          ReductionIdentifier redId,
                                          mlir::Type type, mlir::Value op1,
                                          mlir::Value op2);

  /// Creates an OpenMP reduction declaration and inserts it into the provided
  /// symbol table. The declaration has a constant initializer with the neutral
  /// value `initValue`, and the reduction combiner carried over from `reduce`.
  /// TODO: add atomic region.
  static mlir::omp::DeclareReductionOp
  createDeclareReduction(fir::FirOpBuilder &builder,
                         llvm::StringRef reductionOpName,
                         const ReductionIdentifier redId, mlir::Type type,
                         mlir::Location loc, bool isByRef);

  /// Creates a reduction declaration and associates it with an OpenMP block
  /// directive.
  static void addDeclareReduction(
      mlir::Location currentLocation,
      Fortran::lower::AbstractConverter &converter,
      const omp::clause::Reduction &reduction,
      llvm::SmallVectorImpl<mlir::Value> &reductionVars,
      llvm::SmallVectorImpl<mlir::Attribute> &reductionDeclSymbols,
      llvm::SmallVectorImpl<const Fortran::semantics::Symbol *>
          *reductionSymbols = nullptr);
};

template <typename FloatOp, typename IntegerOp>
mlir::Value
ReductionProcessor::getReductionOperation(fir::FirOpBuilder &builder,
                                          mlir::Type type, mlir::Location loc,
                                          mlir::Value op1, mlir::Value op2) {
  type = fir::unwrapRefType(type);
  assert(type.isIntOrIndexOrFloat() &&
         "only integer and float types are currently supported");
  if (type.isIntOrIndex())
    return builder.create<IntegerOp>(loc, op1, op2);
  return builder.create<FloatOp>(loc, op1, op2);
}

} // namespace omp
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_REDUCTIONPROCESSOR_H
