//===-- Lower/OpenMP.h -- lower Open MP directives --------------*- C++ -*-===//
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

#ifndef FORTRAN_LOWER_OPENMP_H
#define FORTRAN_LOWER_OPENMP_H

#include <cinttypes>

namespace mlir {
class Value;
class Operation;
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
class ConvertOp;
} // namespace fir

namespace Fortran {
namespace parser {
struct OpenMPConstruct;
struct OpenMPDeclarativeConstruct;
struct OmpEndLoopDirective;
struct OmpClauseList;
} // namespace parser

namespace semantics {
class Symbol;
class SemanticsContext;
} // namespace semantics

namespace lower {

class AbstractConverter;
class SymMap;

namespace pft {
struct Evaluation;
struct Variable;
} // namespace pft

// Generate the OpenMP terminator for Operation at Location.
void genOpenMPTerminator(fir::FirOpBuilder &, mlir::Operation *,
                         mlir::Location);

void genOpenMPConstruct(AbstractConverter &, Fortran::lower::SymMap &,
                        semantics::SemanticsContext &, pft::Evaluation &,
                        const parser::OpenMPConstruct &);
void genOpenMPDeclarativeConstruct(AbstractConverter &, pft::Evaluation &,
                                   const parser::OpenMPDeclarativeConstruct &);
/// Symbols in OpenMP code can have flags (e.g. threadprivate directive)
/// that require additional handling when lowering the corresponding
/// variable. Perform such handling according to the flags on the symbol.
/// The variable \p var is required to have a `Symbol`.
void genOpenMPSymbolProperties(AbstractConverter &converter,
                               const pft::Variable &var);

int64_t getCollapseValue(const Fortran::parser::OmpClauseList &clauseList);
void genThreadprivateOp(AbstractConverter &, const pft::Variable &);
void genDeclareTargetIntGlobal(AbstractConverter &, const pft::Variable &);
void genOpenMPReduction(AbstractConverter &,
                        const Fortran::parser::OmpClauseList &clauseList);

mlir::Operation *findReductionChain(mlir::Value, mlir::Value * = nullptr);
fir::ConvertOp getConvertFromReductionOp(mlir::Operation *, mlir::Value);
void updateReduction(mlir::Operation *, fir::FirOpBuilder &, mlir::Value,
                     mlir::Value, fir::ConvertOp * = nullptr);
void removeStoreOp(mlir::Operation *, mlir::Value);

bool isOpenMPTargetConstruct(const parser::OpenMPConstruct &);
bool isOpenMPDeviceDeclareTarget(Fortran::lower::AbstractConverter &,
                                 Fortran::lower::pft::Evaluation &,
                                 const parser::OpenMPDeclarativeConstruct &);
void genOpenMPRequires(mlir::Operation *, const Fortran::semantics::Symbol *);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_OPENMP_H
