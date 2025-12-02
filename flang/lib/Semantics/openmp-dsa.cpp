//===-- flang/lib/Semantics/openmp-dsa.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/openmp-dsa.h"

namespace Fortran::semantics {

Symbol::Flags GetSymbolDSA(const Symbol &symbol) {
  Symbol::Flags dsaFlags{Symbol::Flag::OmpPrivate,
      Symbol::Flag::OmpFirstPrivate, Symbol::Flag::OmpLastPrivate,
      Symbol::Flag::OmpShared, Symbol::Flag::OmpLinear,
      Symbol::Flag::OmpReduction};
  Symbol::Flags dsa{symbol.flags() & dsaFlags};
  if (dsa.any()) {
    return dsa;
  }
  // If no DSA are set use those from the host associated symbol, if any.
  if (const auto *details{symbol.detailsIf<HostAssocDetails>()}) {
    return GetSymbolDSA(details->symbol());
  }
  return {};
}

} // namespace Fortran::semantics
