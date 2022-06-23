//===-- Lower/IntrinsicCall.h -- lowering of intrinsics ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_INTRINSICCALL_H
#define FORTRAN_LOWER_INTRINSICCALL_H

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "llvm/ADT/Optional.h"

namespace fir {
class ExtendedValue;
}

namespace Fortran::lower {

class StatementContext;

// TODO: Error handling interface ?
// TODO: Implementation is incomplete. Many intrinsics to tbd.

/// Generate the FIR+MLIR operations for the generic intrinsic \p name
/// with arguments \p args and expected result type \p resultType.
/// Returned mlir::Value is the returned Fortran intrinsic value.
fir::ExtendedValue genIntrinsicCall(fir::FirOpBuilder &, mlir::Location,
                                    llvm::StringRef name,
                                    llvm::Optional<mlir::Type> resultType,
                                    llvm::ArrayRef<fir::ExtendedValue> args,
                                    StatementContext &);

/// Enum specifying how intrinsic argument evaluate::Expr should be
/// lowered to fir::ExtendedValue to be passed to genIntrinsicCall.
enum class LowerIntrinsicArgAs {
  /// Lower argument to a value. Mainly intended for scalar arguments.
  Value,
  /// Lower argument to an address. Only valid when the argument properties are
  /// fully defined (e.g. allocatable is allocated...).
  Addr,
  /// Lower argument to a box.
  Box,
  /// Lower argument without assuming that the argument is fully defined.
  /// It can be used on unallocated allocatable, disassociated pointer,
  /// or absent optional. This is meant for inquiry intrinsic arguments.
  Inquired
};

/// Define how a given intrinsic argument must be lowered.
struct ArgLoweringRule {
  LowerIntrinsicArgAs lowerAs;
  /// Value:
  //    - Numerical: 0
  //    - Logical : false
  //    - Derived/character: not possible. Need custom intrinsic lowering.
  //  Addr:
  //    - nullptr
  //  Box:
  //    - absent box
  //  AsInquired:
  //    - no-op
  bool handleDynamicOptional;
};

/// Opaque class defining the argument lowering rules for all the argument of
/// an intrinsic.
struct IntrinsicArgumentLoweringRules;

/// Return argument lowering rules for an intrinsic.
/// Returns a nullptr if all the intrinsic arguments should be lowered by value.
const IntrinsicArgumentLoweringRules *
getIntrinsicArgumentLowering(llvm::StringRef intrinsicName);

/// Return how argument \p argName should be lowered given the rules for the
/// intrinsic function. The argument names are the one defined by the standard.
ArgLoweringRule lowerIntrinsicArgumentAs(const IntrinsicArgumentLoweringRules &,
                                         unsigned position);

/// Return place-holder for absent intrinsic arguments.
fir::ExtendedValue getAbsentIntrinsicArgument();

/// Get SymbolRefAttr of runtime (or wrapper function containing inlined
// implementation) of an unrestricted intrinsic (defined by its signature
// and generic name)
mlir::SymbolRefAttr
getUnrestrictedIntrinsicSymbolRefAttr(fir::FirOpBuilder &, mlir::Location,
                                      llvm::StringRef name,
                                      mlir::FunctionType signature);

//===----------------------------------------------------------------------===//
// Direct access to intrinsics that may be used by lowering outside
// of intrinsic call lowering.
//===----------------------------------------------------------------------===//

/// Generate maximum. There must be at least one argument and all arguments
/// must have the same type.
mlir::Value genMax(fir::FirOpBuilder &, mlir::Location,
                   llvm::ArrayRef<mlir::Value> args);

/// Generate minimum. Same constraints as genMax.
mlir::Value genMin(fir::FirOpBuilder &, mlir::Location,
                   llvm::ArrayRef<mlir::Value> args);

/// Generate power function x**y with the given expected
/// result type.
mlir::Value genPow(fir::FirOpBuilder &, mlir::Location, mlir::Type resultType,
                   mlir::Value x, mlir::Value y);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_INTRINSICCALL_H
