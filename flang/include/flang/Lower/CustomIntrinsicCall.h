//===-- Lower/CustomIntrinsicCall.h -----------------------------*- C++ -*-===//
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
/// Custom intrinsic lowering for the few intrinsic that have optional
/// arguments that prevents them to be handled in a more generic way in
/// IntrinsicCall.cpp.
/// The core principle is that this interface provides the intrinsic arguments
/// via callbacks to generate fir::ExtendedValue (instead of a list of
/// precomputed fir::ExtendedValue as done in the default intrinsic call
/// lowering). This gives more flexibility to only generate references to
/// dynamically optional arguments (pointers, allocatables, OPTIONAL dummies) in
/// a safe way.
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CUSTOMINTRINSICCALL_H
#define FORTRAN_LOWER_CUSTOMINTRINSICCALL_H

#include "flang/Lower/AbstractConverter.h"
#include "llvm/ADT/Optional.h"
#include <functional>

namespace Fortran {

namespace evaluate {
class ProcedureRef;
struct SpecificIntrinsic;
} // namespace evaluate

namespace lower {

/// Does the call \p procRef to \p intrinsic need to be handle via this custom
/// framework due to optional arguments. Otherwise, the tools from
/// IntrinsicCall.cpp should be used directly.
bool intrinsicRequiresCustomOptionalHandling(
    const Fortran::evaluate::ProcedureRef &procRef,
    const Fortran::evaluate::SpecificIntrinsic &intrinsic,
    AbstractConverter &converter);

/// Type of callback to be provided to prepare the arguments fetching from an
/// actual argument expression.
using OperandPrepare = std::function<void(const Fortran::lower::SomeExpr &)>;

/// Type of the callback to inquire about an argument presence, once the call
/// preparation was done. An absent optional means the argument is statically
/// present. An mlir::Value means the presence must be checked at runtime, and
/// that the value contains the "is present" boolean value.
using OperandPresent = std::function<llvm::Optional<mlir::Value>(std::size_t)>;

/// Type of the callback to generate an argument reference after the call
/// preparation was done. For optional arguments, the utility guarantees
/// these callbacks will only be called in regions where the presence was
/// verified. This means the getter callback can dereference the argument
/// without any special care.
/// For elemental intrinsics, the getter must provide the current iteration
/// element value.
using OperandGetter = std::function<fir::ExtendedValue(std::size_t)>;

/// Given a callback \p prepareOptionalArgument to prepare optional
/// arguments and a callback \p prepareOtherArgument to prepare non-optional
/// arguments prepare the intrinsic arguments calls.
/// It is up to the caller to decide what argument preparation means,
/// the only contract is that it should later allow the caller to provide
/// callbacks to generate argument reference given an argument index without
/// any further knowledge of the argument. The function simply visits
/// the actual arguments, deciding which ones are dynamically optional,
/// and calling the callbacks accordingly in argument order.
void prepareCustomIntrinsicArgument(
    const Fortran::evaluate::ProcedureRef &procRef,
    const Fortran::evaluate::SpecificIntrinsic &intrinsic,
    llvm::Optional<mlir::Type> retTy,
    const OperandPrepare &prepareOptionalArgument,
    const OperandPrepare &prepareOtherArgument, AbstractConverter &converter);

/// Given a callback \p getOperand to generate a reference to the i-th argument,
/// and a callback \p isPresentCheck to test if an argument is present, this
/// function lowers the intrinsic calls to \p name whose argument were
/// previously prepared with prepareCustomIntrinsicArgument. The elemental
/// aspects must be taken into account by the caller (i.e, the function should
/// be called during the loop nest generation for elemental intrinsics. It will
/// not generate any implicit loop nest on its own).
fir::ExtendedValue
lowerCustomIntrinsic(fir::FirOpBuilder &builder, mlir::Location loc,
                     llvm::StringRef name, llvm::Optional<mlir::Type> retTy,
                     const OperandPresent &isPresentCheck,
                     const OperandGetter &getOperand, std::size_t numOperands,
                     Fortran::lower::StatementContext &stmtCtx);
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_CUSTOMINTRINSICCALL_H
