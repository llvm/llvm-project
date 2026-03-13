//===- FortranVariableInfo.h - Fortran variable info for OpenACC -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fortran-specific variable information carried through the OpenACC type
// interface helpers (generatePrivateInit, generateCopy,
// generatePrivateDestroy). This allows recovering properties such as
// OPTIONAL that are not representable in the FIR type system.
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_OPTIMIZER_OPENACC_FORTRANVARIABLEINFO_H
#define FLANG_OPTIMIZER_OPENACC_FORTRANVARIABLEINFO_H

#include "mlir/Dialect/OpenACC/OpenACCVariableInfo.h"

namespace fir::acc {

class FortranVariableInfo : public mlir::acc::VariableInfoBase {
public:
  explicit FortranVariableInfo(bool mayBeOptional)
      : VariableInfoBase(Language::Fortran), mayBeOptional_(mayBeOptional) {}

  bool getMayBeOptional() const { return mayBeOptional_; }

  static bool classof(const VariableInfoBase *base) {
    return base->getLanguage() == Language::Fortran;
  }

private:
  bool mayBeOptional_;
};

} // namespace fir::acc

#endif // FLANG_OPTIMIZER_OPENACC_FORTRANVARIABLEINFO_H
