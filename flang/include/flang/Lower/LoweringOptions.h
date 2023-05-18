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

#include "flang/Common/MathOptionsBase.h"

namespace Fortran::lower {

class LoweringOptionsBase {
public:
#define LOWERINGOPT(Name, Bits, Default) unsigned Name : Bits;
#define ENUM_LOWERINGOPT(Name, Type, Bits, Default)
#include "flang/Lower/LoweringOptions.def"

protected:
#define LOWERINGOPT(Name, Bits, Default)
#define ENUM_LOWERINGOPT(Name, Type, Bits, Default) unsigned Name : Bits;
#include "flang/Lower/LoweringOptions.def"
};

class LoweringOptions : public LoweringOptionsBase {

public:
#define LOWERINGOPT(Name, Bits, Default)
#define ENUM_LOWERINGOPT(Name, Type, Bits, Default)                            \
  Type get##Name() const { return static_cast<Type>(Name); }                   \
  LoweringOptions &set##Name(Type Value) {                                     \
    Name = static_cast<unsigned>(Value);                                       \
    return *this;                                                              \
  }
#include "flang/Lower/LoweringOptions.def"

  LoweringOptions();

  const Fortran::common::MathOptionsBase &getMathOptions() const {
    return MathOptions;
  }

  Fortran::common::MathOptionsBase &getMathOptions() { return MathOptions; }

private:
  /// Options for handling/optimizing mathematical computations.
  Fortran::common::MathOptionsBase MathOptions;
};

} // namespace Fortran::lower

#endif // FLANG_LOWER_LOWERINGOPTIONS_H
