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

#include "flang/Support/FPMaxminBehavior.h"
#include "flang/Support/MathOptionsBase.h"
#include <cstdint>

namespace Fortran::lower {

/// Initialization mode for automatic (local) variables without explicit
/// or default initialization, selected via -finit-local=.
///
/// Zero and Hex fill every storage byte including struct padding and
/// CHARACTER storage.  QNaN and SNaN initialize each typed field
/// individually; struct padding is not yet covered for those modes
/// (TODO: use whole-struct memset once PR #159788 lands).
enum class InitLocalKind {
  Off,  ///< No initialization (default)
  Zero, ///< Fill with 0x00 bytes (all types, all storage including padding)
  Hex,  ///< Fill with a user-supplied byte pattern (all types, all storage including padding)
  QNaN, ///< Quiet NaN for FP fields; 0xAA byte-splat for non-FP fields (struct padding not yet covered)
  SNaN, ///< Signalling NaN for FP fields; 0xAA byte-splat for non-FP fields (struct padding not yet covered)
};

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

  /// Returns the byte pattern used for -finit-local=0x<hex>.
  uint8_t getInitLocalPattern() const { return InitLocalPattern; }
  LoweringOptions &setInitLocalPattern(uint8_t V) {
    InitLocalPattern = V;
    return *this;
  }

private:
  /// Byte pattern for -finit-local=0x<hex>. Only meaningful when
  /// getInitLocalMode() == InitLocalKind::Hex.
  uint8_t InitLocalPattern = 0;

  /// Options for handling/optimizing mathematical computations.
  Fortran::common::MathOptionsBase MathOptions;
};

} // namespace Fortran::lower

#endif // FLANG_LOWER_LOWERINGOPTIONS_H
