//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_COMMON_REGISTERINFO_X86_H
#define LLDB_REGISTER_COMMON_REGISTERINFO_X86_H
#include "lldb/Register/Arch/lldb-x86-register-enums.h"
#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/ConstString.h"
#include <cstdint>
#include <cstring>
#include "llvm/TargetParser/Triple.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/Support/Compiler.h"

namespace lldb_private {
enum class X86RegisterABI { SystemV, Windows };
enum class X86RegisterMode { Target, I386, X86_64 };

/// Logical register classes exposed by a value profile. No storage choices.
struct X86RegisterProfile {
  bool gpr_aliases = true;
  bool segments = true;
  bool x87 = true;
  bool simd = true;
  bool avx = true;
  bool mpx = true;
  bool debug = true;
};

/// Common x86 value descriptions and feature/ABI variants. Native buffer
/// structures and native API calls belong to the supplying format adapter.
class RegisterInfo_x86 {
public:
  static std::shared_ptr<const RegisterDescription>
  CreateDescription(const ArchSpec &arch, bool with_segment_bases = false,
                    X86RegisterABI abi = X86RegisterABI::SystemV,
                    X86RegisterMode mode = X86RegisterMode::Target,
                    X86RegisterProfile profile = {});
};
// Generic floating-point registers

LLVM_PACKED_START
struct MMSRegComp {
  uint64_t mantissa;
  uint16_t sign_exp;
};

struct MMSReg {
  union {
    uint8_t bytes[10];
    MMSRegComp comp;
  };
  uint8_t pad[6];
};
LLVM_PACKED_END

static_assert(sizeof(MMSRegComp) == 10, "MMSRegComp is not 10 bytes of size");
static_assert(sizeof(MMSReg) == 16, "MMSReg is not 16 bytes of size");

struct XMMReg {
  uint8_t bytes[16]; // 128-bits for each XMM register
};

// i387_fxsave_struct
struct FXSAVE {
  uint16_t fctrl;     // FPU Control Word (fcw)
  uint16_t fstat;     // FPU Status Word (fsw)
  uint16_t ftag;      // FPU Tag Word (ftw)
  uint16_t fop;       // Last Instruction Opcode (fop)
  union {
    struct {
      uint64_t fip; // Instruction Pointer
      uint64_t fdp; // Data Pointer
    } x86_64;
    struct {
      uint32_t fioff; // FPU IP Offset (fip)
      uint32_t fiseg; // FPU IP Selector (fcs)
      uint32_t fooff; // FPU Operand Pointer Offset (foo)
      uint32_t foseg; // FPU Operand Pointer Selector (fos)
    } i386_; // Added _ in the end to avoid error with gcc defining i386 in some
             // cases
  } ptr;
  uint32_t mxcsr;     // MXCSR Register State
  uint32_t mxcsrmask; // MXCSR Mask
  MMSReg stmm[8];     // 8*16 bytes for each FP-reg = 128 bytes
  XMMReg xmm[16];     // 16*16 bytes for each XMM-reg = 256 bytes
  uint8_t padding1[48];
  uint64_t xcr0;
  uint8_t padding2[40];
};

// Extended floating-point registers

struct YMMHReg {
  uint8_t bytes[16]; // 16 * 8 bits for the high bytes of each YMM register
};

struct YMMReg {
  uint8_t bytes[32]; // 16 * 16 bits for each YMM register
};

struct YMM {
  YMMReg ymm[16]; // assembled from ymmh and xmm registers
};

struct MPXReg {
  uint8_t bytes[16]; // MPX 128 bit bound registers
};

struct MPXCsr {
  uint8_t bytes[8]; // MPX 64 bit bndcfgu and bndstatus registers (collectively
                    // BNDCSR state)
};

struct MPX {
  MPXReg mpxr[4];
  MPXCsr mpxc[2];
};

LLVM_PACKED_START
struct XSAVE_HDR {
  enum class XFeature : uint64_t {
    FP = 1,
    SSE = FP << 1,
    YMM = SSE << 1,
    BNDREGS = YMM << 1,
    BNDCSR = BNDREGS << 1,
    OPMASK = BNDCSR << 1,
    ZMM_Hi256 = OPMASK << 1,
    Hi16_ZMM = ZMM_Hi256 << 1,
    PT = Hi16_ZMM << 1,
    PKRU = PT << 1,
    LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue*/ PKRU)
  };

  XFeature xstate_bv; // OS enabled xstate mask to determine the extended states
                      // supported by the processor
  XFeature xcomp_bv;  // Mask to indicate the format of the XSAVE area and of
                      // the XRSTOR instruction
  uint64_t reserved1[1];
  uint64_t reserved2[5];
};
static_assert(sizeof(XSAVE_HDR) == 64, "XSAVE_HDR layout incorrect");
LLVM_PACKED_END

// x86 extensions to FXSAVE (i.e. for AVX and MPX processors)
LLVM_PACKED_START
struct XSAVE {
  FXSAVE i387;      // floating point registers typical in i387_fxsave_struct
  XSAVE_HDR header; // The xsave_hdr_struct can be used to determine if the
                    // following extensions are usable
  YMMHReg ymmh[16]; // High 16 bytes of each of 16 YMM registers (the low bytes
                    // are in FXSAVE.xmm for compatibility with SSE)
  uint64_t reserved3[16];
  MPXReg mpxr[4];   // MPX BNDREG state, containing 128-bit bound registers
  MPXCsr mpxc[2];   // MPX BNDCSR state, containing 64-bit BNDCFGU and
                    // BNDSTATUS registers
};
LLVM_PACKED_END

// Floating-point registers
union FPR {
  FXSAVE fxsave; // Generic floating-point registers.
  XSAVE xsave;   // x86 extended processor state.
};

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

// Convenience function to combine YMM register data from XSAVE-style input.
inline YMMReg XStateToYMM(const void* xmm_bytes, const void* ymmh_bytes) {
  YMMReg ret;

  ::memcpy(ret.bytes, xmm_bytes, sizeof(XMMReg));
  ::memcpy(ret.bytes + sizeof(XMMReg), ymmh_bytes, sizeof(YMMHReg));

  return ret;
}

// Convenience function to copy YMM register data into XSAVE-style output.
inline void YMMToXState(const YMMReg& input, void* xmm_bytes, void* ymmh_bytes) {
  ::memcpy(xmm_bytes, input.bytes, sizeof(XMMReg));
  ::memcpy(ymmh_bytes, input.bytes + sizeof(XMMReg), sizeof(YMMHReg));
}

uint16_t AbridgedToFullTagWord(uint8_t abridged_tw, uint16_t sw,
                               llvm::ArrayRef<MMSReg> st_regs);
uint8_t FullToAbridgedTagWord(uint16_t tw);

namespace x86_layout {
enum Bank : RegisterBankID { GP, FP, YMMHigh, Bounds, BoundsControl, Debug };
}
/// OS adapters supply their GPR fields; hardware register formats are shared.
RegisterLayout CreateX86RegisterLayout(
    const ArchSpec &arch, llvm::ArrayRef<RegisterStorageField> gpr_fields,
    size_t gpr_size, bool gpr_is_64, bool segment_bases = false,
    X86RegisterABI abi = X86RegisterABI::SystemV);
} // namespace lldb_private
#endif
