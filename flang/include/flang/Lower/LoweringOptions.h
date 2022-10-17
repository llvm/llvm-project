//===- LoweringOptions.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Options controlling lowering of front-end fragments to the FIR dialect
/// of MLIR
///
//===----------------------------------------------------------------------===//

#ifndef FLANG_LOWER_LOWERINGOPTIONS_H
#define FLANG_LOWER_LOWERINGOPTIONS_H

namespace Fortran::lower {

class LoweringOptions {
  /// If true, lower transpose without a runtime call.
  unsigned optimizeTranspose : 1;

  /// If true, enable polymorphic type lowering feature. Off by default.
  unsigned polymorphicTypeImpl : 1;

  /// If true, lower to High level FIR before lowering to FIR.
  /// Off by default until fully ready.
  unsigned lowerToHighLevelFIR : 1;

public:
  LoweringOptions()
      : optimizeTranspose(true), polymorphicTypeImpl(false),
        lowerToHighLevelFIR(false) {}

  bool getOptimizeTranspose() const { return optimizeTranspose; }
  LoweringOptions &setOptimizeTranspose(bool v) {
    optimizeTranspose = v;
    return *this;
  }

  bool isPolymorphicTypeImplEnabled() const { return polymorphicTypeImpl; }
  LoweringOptions &setPolymorphicTypeImpl(bool v) {
    polymorphicTypeImpl = v;
    return *this;
  }

  bool getLowerToHighLevelFIR() const { return lowerToHighLevelFIR; }
  LoweringOptions &setLowerToHighLevelFIR(bool v) {
    lowerToHighLevelFIR = v;
    return *this;
  }
};

} // namespace Fortran::lower

#endif // FLANG_LOWER_LOWERINGOPTIONS_H
