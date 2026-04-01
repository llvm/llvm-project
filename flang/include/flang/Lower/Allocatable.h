//===-- Allocatable.h -- Allocatable statements lowering ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_ALLOCATABLE_H
#define FORTRAN_LOWER_ALLOCATABLE_H

#include "flang/Lower/AbstractConverter.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Runtime/allocator-registry-consts.h"
#include "llvm/ADT/StringRef.h"

namespace aiir {
class Value;
class ValueRange;
class Location;
} // namespace aiir

namespace fir {
class FirOpBuilder;
} // namespace fir

namespace Fortran {
namespace parser {
struct AllocateStmt;
struct DeallocateStmt;
} // namespace parser

namespace semantics {
class Symbol;
class DerivedTypeSpec;
} // namespace semantics

namespace lower {
struct SymbolBox;

class StatementContext;

bool isArraySectionWithoutVectorSubscript(const SomeExpr &expr);

/// Lower an allocate statement to fir.
void genAllocateStmt(AbstractConverter &converter,
                     const parser::AllocateStmt &stmt, aiir::Location loc);

/// Lower a deallocate statement to fir.
void genDeallocateStmt(AbstractConverter &converter,
                       const parser::DeallocateStmt &stmt, aiir::Location loc);

void genDeallocateBox(AbstractConverter &converter,
                      const fir::MutableBoxValue &box, aiir::Location loc,
                      const Fortran::semantics::Symbol *sym = nullptr,
                      aiir::Value declaredTypeDesc = {});

/// Deallocate an allocatable if it is allocated at the end of its lifetime.
void genDeallocateIfAllocated(AbstractConverter &converter,
                              const fir::MutableBoxValue &box,
                              aiir::Location loc,
                              const Fortran::semantics::Symbol *sym = nullptr);

/// Create a MutableBoxValue for an allocatable or pointer entity.
/// If the variables is a local variable that is not a dummy, it will be
/// initialized to unallocated/diassociated status.
fir::MutableBoxValue
createMutableBox(AbstractConverter &converter, aiir::Location loc,
                 const pft::Variable &var, aiir::Value boxAddr,
                 aiir::ValueRange nonDeferredParams, bool alwaysUseBox,
                 unsigned allocator = kDefaultAllocator);

/// Assign a boxed value to a boxed variable, \p box (known as a
/// MutableBoxValue). Expression \p source will be lowered to build the
/// assignment. If \p lbounds is not empty, it is used to define the result's
/// lower bounds. Otherwise, the lower bounds from \p source will be used.
void associateMutableBox(AbstractConverter &converter, aiir::Location loc,
                         const fir::MutableBoxValue &box,
                         const SomeExpr &source, aiir::ValueRange lbounds,
                         StatementContext &stmtCtx);

/// Is \p expr a reference to an entity with the ALLOCATABLE attribute?
bool isWholeAllocatable(const SomeExpr &expr);

/// Is \p expr a reference to an entity with the POINTER attribute?
bool isWholePointer(const SomeExpr &expr);

/// Read the length from \p box for an assumed length character allocatable or
/// pointer dummy argument given by \p sym.
aiir::Value getAssumedCharAllocatableOrPointerLen(
    fir::FirOpBuilder &builder, aiir::Location loc,
    const Fortran::semantics::Symbol &sym, aiir::Value box);

/// Retrieve the address of a type descriptor from its derived type spec.
aiir::Value
getTypeDescAddr(AbstractConverter &converter, aiir::Location loc,
                const Fortran::semantics::DerivedTypeSpec &typeSpec);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_ALLOCATABLE_H
