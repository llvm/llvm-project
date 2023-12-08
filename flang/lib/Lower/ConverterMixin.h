//===-- ConverterMixin.h --------------------------------------------------===//
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

#ifndef FORTRAN_LOWER_CONVERTERMIXIN_H
#define FORTRAN_LOWER_CONVERTERMIXIN_H

namespace Fortran::lower {

template <typename FirConverterT> class ConverterMixinBase {
public:
  FirConverterT *This() { return static_cast<FirConverterT *>(this); }
  const FirConverterT *This() const {
    return static_cast<const FirConverterT *>(this);
  }
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_CONVERTERMIXIN_H
