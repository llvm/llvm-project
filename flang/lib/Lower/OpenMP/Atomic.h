//===-- Atomic.h -- Lowering of atomic constructs -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef FORTRAN_LOWER_OPENMP_ATOMIC_H
#define FORTRAN_LOWER_OPENMP_ATOMIC_H

namespace Fortran {
namespace lower {
class AbstractConverter;
class SymMap;

namespace pft {
struct Evaluation;
}
} // namespace lower

namespace parser {
struct OpenMPAtomicConstruct;
}

namespace semantics {
class SemanticsContext;
}
} // namespace Fortran

namespace Fortran::lower::omp {
void lowerAtomic(AbstractConverter &converter, SymMap &symTable,
                 semantics::SemanticsContext &semaCtx, pft::Evaluation &eval,
                 const parser::OpenMPAtomicConstruct &construct);
}

#endif // FORTRAN_LOWER_OPENMP_ATOMIC_H
