//===-- flang/lib/Semantics/openmp-dsa.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/openmp-dsa.h"

namespace Fortran::semantics {

Symbol::Flags GetDataSharingAttributeFlags() {
  return Symbol::Flags{Symbol::Flag::OmpShared, Symbol::Flag::OmpPrivate,
      Symbol::Flag::OmpFirstPrivate, Symbol::Flag::OmpLastPrivate,
      Symbol::Flag::OmpReduction, Symbol::Flag::OmpLinear};
}

Symbol::Flags GetSymbolDSA(const Symbol &symbol) {
  Symbol::Flags dsa{symbol.flags() & GetDataSharingAttributeFlags()};
  if (dsa.any()) {
    return dsa;
  }
  // If no DSA are set use those from the host associated symbol, if any.
  if (const auto *details{symbol.detailsIf<HostAssocDetails>()}) {
    return GetSymbolDSA(details->symbol());
  }
  return {};
}

// Clear any previous data-sharing attribute flags and set the new ones.
// Needed when setting PreDetermined DSAs, that take precedence over Implicit
// ones.
void SetSymbolDSA(Symbol &symbol, Symbol::Flags flags) {
  symbol.flags() &= ~(GetDataSharingAttributeFlags() |
      Symbol::Flags{Symbol::Flag::OmpExplicit, Symbol::Flag::OmpImplicit,
          Symbol::Flag::OmpPreDetermined});
  symbol.flags() |= flags;
}

} // namespace Fortran::semantics
