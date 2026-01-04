//===-- include/flang/Semantics/openmp-dsa.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_OPENMP_DSA_H_
#define FORTRAN_SEMANTICS_OPENMP_DSA_H_

#include "flang/Semantics/symbol.h"

namespace Fortran::semantics {

Symbol::Flags GetSymbolDSA(const Symbol &symbol);

} // namespace Fortran::semantics

#endif // FORTRAN_SEMANTICS_OPENMP_DSA_H_
