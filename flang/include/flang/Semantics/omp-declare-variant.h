//===-- flang/Semantics/omp-declare-variant.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_OMP_DECLARE_VARIANT_H_
#define FORTRAN_SEMANTICS_OMP_DECLARE_VARIANT_H_

#include "flang/Common/reference.h"

namespace Fortran::parser {
inline namespace traits {
struct OmpContextSelectorSpecification;
} // namespace traits
} // namespace Fortran::parser

namespace Fortran::semantics {
class Symbol;

// Information about a DECLARE VARIANT directive that will be recorded on its
// base procedure.
struct OmpDeclareVariantEntry {
  common::Reference<const Symbol> variant;
  const parser::traits::OmpContextSelectorSpecification *matchSelector{nullptr};
};

} // namespace Fortran::semantics

#endif // FORTRAN_SEMANTICS_OMP_DECLARE_VARIANT_H_
