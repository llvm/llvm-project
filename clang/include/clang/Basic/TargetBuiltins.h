//===--- TargetBuiltins.h - Target specific builtin IDs ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Enumerates target-specific builtins in their own namespaces within
/// namespace ::clang.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_TARGETBUILTINS_H
#define LLVM_CLANG_BASIC_TARGETBUILTINS_H

#include <stdint.h>
#include "clang/Basic/Builtins.h"
#include "llvm/Support/MathExtras.h"
#undef PPC

namespace clang {

  namespace NEON {
  enum {
    LastTIBuiltin = clang::Builtin::FirstTSBuiltin - 1,
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "clang/Basic/BuiltinsNEON.def"
    FirstTSBuiltin
  };
  }

  /// ARM builtins
  namespace ARM {
    enum {
      LastTIBuiltin = clang::Builtin::FirstTSBuiltin-1,
      LastNEONBuiltin = NEON::FirstTSBuiltin - 1,
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "clang/Basic/BuiltinsARM.def"
      LastTSBuiltin
    };
  }

  namespace SVE {
  enum {
    LastNEONBuiltin = NEON::FirstTSBuiltin - 1,
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "clang/Basic/BuiltinsSVE.def"
    FirstTSBuiltin,
  };
  }

  /// AArch64 builtins
  namespace AArch64 {
  enum {
    LastTIBuiltin = clang::Builtin::FirstTSBuiltin - 1,
    LastNEONBuiltin = NEON::FirstTSBuiltin - 1,
    FirstSVEBuiltin = NEON::FirstTSBuiltin,
    LastSVEBuiltin = SVE::FirstTSBuiltin - 1,
  #define BUILTIN(ID, TYPE, ATTRS) BI##ID,
  #include "clang/Basic/BuiltinsAArch64.def"
    LastTSBuiltin
  };
  }

  /// BPF builtins
  namespace BPF {
  enum {
    LastTIBuiltin = clang::Builtin::FirstTSBuiltin - 1,
  #define BUILTIN(ID, TYPE, ATTRS) BI##ID,
  #include "clang/Basic/BuiltinsBPF.def"
    LastTSBuiltin
  };
  }

  /// PPC builtins
  namespace PPC {
    enum {
        LastTIBuiltin = clang::Builtin::FirstTSBuiltin-1,
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "clang/Basic/BuiltinsPPC.def"
        LastTSBuiltin
    };
  }

  /// NVPTX builtins
  namespace NVPTX {
    enum {
        LastTIBuiltin = clang::Builtin::FirstTSBuiltin-1,
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "clang/Basic/BuiltinsNVPTX.def"
        LastTSBuiltin
    };
  }

  /// AMDGPU builtins
  namespace AMDGPU {
  enum {
    LastTIBuiltin = clang::Builtin::FirstTSBuiltin - 1,
  #define BUILTIN(ID, TYPE, ATTRS) BI##ID,
  #include "clang/Basic/BuiltinsAMDGPU.def"
    LastTSBuiltin
  };
  }

  /// X86 builtins
  namespace X86 {
  enum {
    LastTIBuiltin = clang::Builtin::FirstTSBuiltin - 1,
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "clang/Basic/BuiltinsX86.def"
    FirstX86_64Builtin,
    LastX86CommonBuiltin = FirstX86_64Builtin - 1,
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "clang/Basic/BuiltinsX86_64.def"
    LastTSBuiltin
  };
  }

  /// Flags to identify the types for overloaded Neon builtins.
  ///
  /// These must be kept in sync with the flags in utils/TableGen/NeonEmitter.h.
  class NeonTypeFlags {
    enum {
      EltTypeMask = 0xf,
      UnsignedFlag = 0x10,
      QuadFlag = 0x20
    };
    uint32_t Flags;

  public:
    enum EltType {
      Int8,
      Int16,
      Int32,
      Int64,
      Poly8,
      Poly16,
      Poly64,
      Poly128,
      Float16,
      Float32,
      Float64
    };

    NeonTypeFlags(unsigned F) : Flags(F) {}
    NeonTypeFlags(EltType ET, bool IsUnsigned, bool IsQuad) : Flags(ET) {
      if (IsUnsigned)
        Flags |= UnsignedFlag;
      if (IsQuad)
        Flags |= QuadFlag;
    }

    EltType getEltType() const { return (EltType)(Flags & EltTypeMask); }
    bool isPoly() const {
      EltType ET = getEltType();
      return ET == Poly8 || ET == Poly16;
    }
    bool isUnsigned() const { return (Flags & UnsignedFlag) != 0; }
    bool isQuad() const { return (Flags & QuadFlag) != 0; }
  };

  /// Flags to identify the types for overloaded SVE builtins.
  class SVETypeFlags {
    uint64_t Flags;
    unsigned EltTypeShift;
    unsigned MemEltTypeShift;
    unsigned MergeTypeShift;

  public:
#define LLVM_GET_SVE_TYPEFLAGS
#include "clang/Basic/arm_sve_typeflags.inc"
#undef LLVM_GET_SVE_TYPEFLAGS

    enum EltType {
#define LLVM_GET_SVE_ELTTYPES
#include "clang/Basic/arm_sve_typeflags.inc"
#undef LLVM_GET_SVE_ELTTYPES
    };

    enum MemEltType {
#define LLVM_GET_SVE_MEMELTTYPES
#include "clang/Basic/arm_sve_typeflags.inc"
#undef LLVM_GET_SVE_MEMELTTYPES
    };

    enum MergeType {
#define LLVM_GET_SVE_MERGETYPES
#include "clang/Basic/arm_sve_typeflags.inc"
#undef LLVM_GET_SVE_MERGETYPES
    };

    enum ImmCheckType {
#define LLVM_GET_SVE_IMMCHECKTYPES
#include "clang/Basic/arm_sve_typeflags.inc"
#undef LLVM_GET_SVE_IMMCHECKTYPES
    };

    SVETypeFlags(uint64_t F) : Flags(F) {
      EltTypeShift = llvm::countTrailingZeros(EltTypeMask);
      MemEltTypeShift = llvm::countTrailingZeros(MemEltTypeMask);
      MergeTypeShift = llvm::countTrailingZeros(MergeTypeMask);
    }

    EltType getEltType() const {
      return (EltType)((Flags & EltTypeMask) >> EltTypeShift);
    }

    MemEltType getMemEltType() const {
      return (MemEltType)((Flags & MemEltTypeMask) >> MemEltTypeShift);
    }

    MergeType getMergeType() const {
      return (MergeType)((Flags & MergeTypeMask) >> MergeTypeShift);
    }

    bool isLoad() const { return Flags & IsLoad; }
    bool isStore() const { return Flags & IsStore; }
    bool isGatherLoad() const { return Flags & IsGatherLoad; }
    bool isScatterStore() const { return Flags & IsScatterStore; }
    bool isStructLoad() const { return Flags & IsStructLoad; }
    bool isStructStore() const { return Flags & IsStructStore; }
    bool isZExtReturn() const { return Flags & IsZExtReturn; }

    uint64_t getBits() const { return Flags; }
    bool isFlagSet(uint64_t Flag) const { return Flags & Flag; }
  };

  /// Hexagon builtins
  namespace Hexagon {
    enum {
        LastTIBuiltin = clang::Builtin::FirstTSBuiltin-1,
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "clang/Basic/BuiltinsHexagon.def"
        LastTSBuiltin
    };
  }

  /// MIPS builtins
  namespace Mips {
    enum {
        LastTIBuiltin = clang::Builtin::FirstTSBuiltin-1,
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "clang/Basic/BuiltinsMips.def"
        LastTSBuiltin
    };
  }

  /// XCore builtins
  namespace XCore {
    enum {
        LastTIBuiltin = clang::Builtin::FirstTSBuiltin-1,
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "clang/Basic/BuiltinsXCore.def"
        LastTSBuiltin
    };
  }

  /// Le64 builtins
  namespace Le64 {
  enum {
    LastTIBuiltin = clang::Builtin::FirstTSBuiltin - 1,
  #define BUILTIN(ID, TYPE, ATTRS) BI##ID,
  #include "clang/Basic/BuiltinsLe64.def"
    LastTSBuiltin
  };
  }

  /// SystemZ builtins
  namespace SystemZ {
    enum {
        LastTIBuiltin = clang::Builtin::FirstTSBuiltin-1,
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "clang/Basic/BuiltinsSystemZ.def"
        LastTSBuiltin
    };
  }

  /// WebAssembly builtins
  namespace WebAssembly {
    enum {
      LastTIBuiltin = clang::Builtin::FirstTSBuiltin-1,
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "clang/Basic/BuiltinsWebAssembly.def"
      LastTSBuiltin
    };
  }

} // end namespace clang.

#endif
