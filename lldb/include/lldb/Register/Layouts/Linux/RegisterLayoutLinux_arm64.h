//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_ARM64_H
#define LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_ARM64_H
#include "lldb/Register/Common/RegisterInfo_arm64.h"
#include "lldb/Register/RegisterLayout.h"
namespace lldb_private {
namespace sve {

/*
 * The SVE architecture leaves space for future expansion of the
 * vector length beyond its initial architectural limit of 2048 bits
 * (16 quadwords).
 *
 * See <Linux kernel source tree>/Documentation/arm64/sve.rst for a description
 * of the vl/vq terminology.
 */

const uint16_t vq_bytes = 16; /* number of bytes per quadword */

const uint16_t vq_min = 1;
const uint16_t vq_max = 512;

const uint16_t vl_min = vq_min * vq_bytes;
const uint16_t vl_max = vq_max * vq_bytes;

const uint16_t num_of_zregs = 32;
const uint16_t num_of_pregs = 16;

inline uint16_t vl_valid(uint16_t vl) {
  return (vl % vq_bytes == 0 && vl >= vl_min && vl <= vl_max);
}

inline uint16_t vq_from_vl(uint16_t vl) { return vl / vq_bytes; }
inline uint16_t vl_from_vq(uint16_t vq) { return vq * vq_bytes; }

/* A new signal frame record sve_context encodes the SVE Registers on signal
 * delivery. sve_context struct definition may be included in asm/sigcontext.h.
 * We define sve_context_size which will be used by LLDB sve helper functions.
 * More information on sve_context can be found in Linux kernel source tree at
 * Documentation/arm64/sve.rst.
 */

const uint16_t sve_context_size = 16;

/*
 * If the SVE registers are currently live for the thread at signal delivery,
 * sve_context.head.size >=
 * SigContextSize(vq_from_vl(sve_context.vl))
 * and the register data may be accessed using the Sig*() functions.
 *
 * If sve_context.head.size <
 * SigContextSize(vq_from_vl(sve_context.vl)),
 * the SVE registers were not live for the thread and no register data
 * is included: in this case, the Sig*() functions should not be
 * used except for this check.
 *
 * The same convention applies when returning from a signal: a caller
 * will need to remove or resize the sve_context block if it wants to
 * make the SVE registers live when they were previously non-live or
 * vice-versa.  This may require the caller to allocate fresh
 * memory and/or move other context blocks in the signal frame.
 *
 * Changing the vector length during signal return is not permitted:
 * sve_context.vl must equal the thread's current vector length when
 * doing a sigreturn.
 *
 *
 * Note: for all these functions, the "vq" argument denotes the SVE
 * vector length in quadwords (i.e., units of 128 bits).
 *
 * The correct way to obtain vq is to use vq_from_vl(vl).  The
 * result is valid if and only if vl_valid(vl) is true.  This is
 * guaranteed for a struct sve_context written by the kernel.
 *
 *
 * Additional functions describe the contents and layout of the payload.
 * For each, Sig*Offset(args) is the start offset relative to
 * the start of struct sve_context, and Sig*Size(args) is the
 * size in bytes:
 *
 *	x	type				description
 *	-	----				-----------
 *	REGS					the entire SVE context
 *
 *	ZREGS	__uint128_t[num_of_zregs][vq]	all Z-registers
 *	ZREG	__uint128_t[vq]			individual Z-register Zn
 *
 *	PREGS	uint16_t[num_of_pregs][vq]	all P-registers
 *	PREG	uint16_t[vq]			individual P-register Pn
 *
 *	FFR	uint16_t[vq]			first-fault status register
 *
 * Additional data might be appended in the future.
 */

inline uint16_t SigZRegSize(uint16_t vq) { return vq * vq_bytes; }
inline uint16_t SigPRegSize(uint16_t vq) { return vq * vq_bytes / 8; }
inline uint16_t SigFFRSize(uint16_t vq) { return SigPRegSize(vq); }

inline uint32_t SigRegsOffset() {
  return (sve_context_size + vq_bytes - 1) / vq_bytes * vq_bytes;
}

inline uint32_t SigZRegsOffset() { return SigRegsOffset(); }

inline uint32_t SigZRegOffset(uint16_t vq, uint16_t n) {
  return SigRegsOffset() + SigZRegSize(vq) * n;
}

inline uint32_t SigZRegsSize(uint16_t vq) {
  return SigZRegOffset(vq, num_of_zregs) - SigRegsOffset();
}

inline uint32_t SigPRegsOffset(uint16_t vq) {
  return SigRegsOffset() + SigZRegsSize(vq);
}

inline uint32_t SigPRegOffset(uint16_t vq, uint16_t n) {
  return SigPRegsOffset(vq) + SigPRegSize(vq) * n;
}

inline uint32_t SigpRegsSize(uint16_t vq) {
  return SigPRegOffset(vq, num_of_pregs) - SigPRegsOffset(vq);
}

inline uint32_t SigFFROffset(uint16_t vq) {
  return SigPRegsOffset(vq) + SigpRegsSize(vq);
}

inline uint32_t SigRegsSize(uint16_t vq) {
  return SigFFROffset(vq) + SigFFRSize(vq) - SigRegsOffset();
}

inline uint32_t SVESigContextSize(uint16_t vq) {
  return SigRegsOffset() + SigRegsSize(vq);
}

struct user_sve_header {
  uint32_t size;     /* total meaningful regset content in bytes */
  uint32_t max_size; /* maxmium possible size for this thread */
  uint16_t vl;       /* current vector length */
  uint16_t max_vl;   /* maximum possible vector length */
  uint16_t flags;
  uint16_t reserved;
};

using user_za_header = user_sve_header;

/* Definitions for user_sve_header.flags: */
const uint16_t ptrace_regs_mask = 1 << 0;
const uint16_t ptrace_regs_fpsimd = 0;
const uint16_t ptrace_regs_sve = ptrace_regs_mask;

/*
 * The remainder of the SVE state follows struct user_sve_header.  The
 * total size of the SVE state (including header) depends on the
 * metadata in the header:  PTraceSize(vq, flags) gives the total size
 * of the state in bytes, including the header.
 *
 * Refer to <asm/sigcontext.h> for details of how to pass the correct
 * "vq" argument to these macros.
 */

/* Offset from the start of struct user_sve_header to the register data */
inline uint16_t PTraceRegsOffset() {
  return (sizeof(struct user_sve_header) + vq_bytes - 1) / vq_bytes * vq_bytes;
}

/*
 * The register data content and layout depends on the value of the
 * flags field.
 */

/*
 * (flags & ptrace_regs_mask) == ptrace_regs_fpsimd case:
 *
 * The payload starts at offset PTraceFPSIMDOffset, and is of type
 * struct user_fpsimd_state.  Additional data might be appended in the
 * future: use PTraceFPSIMDSize(vq, flags) to compute the total size.
 * PTraceFPSIMDSize(vq, flags) will never be less than
 * sizeof(struct user_fpsimd_state).
 */

const uint32_t ptrace_fpsimd_offset = PTraceRegsOffset();

/* Return size of struct user_fpsimd_state from asm/ptrace.h */
inline uint32_t PTraceFPSIMDSize(uint16_t vq, uint16_t flags) { return 528; }

/*
 * (flags & ptrace_regs_mask) == ptrace_regs_sve case:
 *
 * The payload starts at offset PTraceSVEOffset, and is of size
 * PTraceSVESize(vq, flags).
 *
 * Additional functions describe the contents and layout of the payload.
 * For each, PTrace*X*Offset(args) is the start offset relative to
 * the start of struct user_sve_header, and PTrace*X*Size(args) is
 * the size in bytes:
 *
 *	x	type				description
 *	-	----				-----------
 *	ZREGS		\
 *	ZREG		|
 *	PREGS		| refer to <asm/sigcontext.h>
 *	PREG		|
 *	FFR		/
 *
 *	FPSR	uint32_t			FPSR
 *	FPCR	uint32_t			FPCR
 *
 * Additional data might be appended in the future.
 */

inline uint32_t PTraceZRegSize(uint16_t vq) { return SigZRegSize(vq); }

inline uint32_t PTracePRegSize(uint16_t vq) { return SigPRegSize(vq); }

inline uint32_t PTraceFFRSize(uint16_t vq) { return SigFFRSize(vq); }

const uint32_t fpsr_size = sizeof(uint32_t);
const uint32_t fpcr_size = sizeof(uint32_t);

inline uint32_t SigToPTrace(uint32_t offset) {
  return offset - SigRegsOffset() + PTraceRegsOffset();
}

const uint32_t ptrace_sve_offset = PTraceRegsOffset();

inline uint32_t PTraceZRegsOffset(uint16_t vq) {
  return SigToPTrace(SigZRegsOffset());
}

inline uint32_t PTraceZRegOffset(uint16_t vq, uint16_t n) {
  return SigToPTrace(SigZRegOffset(vq, n));
}

inline uint32_t PTraceZRegsSize(uint16_t vq) {
  return PTraceZRegOffset(vq, num_of_zregs) - SigToPTrace(SigRegsOffset());
}

inline uint32_t PTracePRegsOffset(uint16_t vq) {
  return SigToPTrace(SigPRegsOffset(vq));
}

inline uint32_t PTracePRegOffset(uint16_t vq, uint16_t n) {
  return SigToPTrace(SigPRegOffset(vq, n));
}

inline uint32_t PTracePRegsSize(uint16_t vq) {
  return PTracePRegOffset(vq, num_of_pregs) - PTracePRegsOffset(vq);
}

inline uint32_t PTraceFFROffset(uint16_t vq) {
  return SigToPTrace(SigFFROffset(vq));
}

inline uint32_t PTraceFPSROffset(uint16_t vq) {
  return (PTraceFFROffset(vq) + PTraceFFRSize(vq) + (vq_bytes - 1)) / vq_bytes *
         vq_bytes;
}

inline uint32_t PTraceFPCROffset(uint16_t vq) {
  return PTraceFPSROffset(vq) + fpsr_size;
}

/*
 * Any future extension appended after FPCR must be aligned to the next
 * 128-bit boundary.
 */

inline uint32_t PTraceSVESize(uint16_t vq, uint16_t flags) {
  return (PTraceFPCROffset(vq) + fpcr_size - ptrace_sve_offset + vq_bytes - 1) /
         vq_bytes * vq_bytes;
}

inline uint32_t PTraceSize(uint16_t vq, uint16_t flags) {
  return (flags & ptrace_regs_mask) == ptrace_regs_sve
             ? ptrace_sve_offset + PTraceSVESize(vq, flags)
             : ptrace_fpsimd_offset + PTraceFPSIMDSize(vq, flags);
}

} // namespace sve
} // namespace lldb_private

namespace lldb_private {
namespace linux_arm64 {
struct GPR {
  uint64_t x[33];
  uint32_t cpsr;
  uint32_t pad;
};
struct alignas(16) FPU {
  uint8_t v[32][16];
  uint32_t fpsr;
  uint32_t fpcr;
};
struct PAuthMasks {
  uint64_t data_mask;
  uint64_t code_mask;
};
struct TLSRegisters {
  uint64_t tpidr;
  uint64_t tpidr2;
};
struct MTEControl {
  uint64_t tagged_addr_ctrl;
};
struct SMERegisters {
  uint64_t ctrl_reg;
  uint64_t svg_reg;
};
struct ZTRegister {
  uint8_t zt0[64];
};
struct FPMRRegister {
  uint64_t fpmr;
};
struct GCSRegisters {
  uint64_t features_enabled;
  uint64_t features_locked;
  uint64_t gcspr_el0;
};
struct POERegister {
  uint64_t por_el0;
};
static_assert(sizeof(GPR) == 272);
static_assert(sizeof(FPU) == 528);
static_assert(sizeof(PAuthMasks) == 16);
static_assert(sizeof(TLSRegisters) == 16);
static_assert(sizeof(MTEControl) == 8);
static_assert(sizeof(SMERegisters) == 16);
static_assert(sizeof(ZTRegister) == 64);
static_assert(sizeof(FPMRRegister) == 8);
static_assert(sizeof(GCSRegisters) == 24);
static_assert(sizeof(POERegister) == 8);
enum Bank : RegisterBankID {
  GP,
  FP,
  SVE,
  PAC,
  MTE,
  TLS,
  SMEControl,
  ZA,
  ZT,
  FPMR,
  GCS,
  POE
};
struct RegisterConfig {
  Flags features;
  uint64_t hwcap = 0;
  uint64_t hwcap2 = 0;
  uint64_t hwcap3 = 0;
  uint32_t sve_vq = 0;
  uint32_t za_vq = 0;
  SVEState sve_state = SVEState::Full;
};
} // namespace linux_arm64
Arm64RegisterFeatures DecodeLinuxArm64RegisterFeatures(uint64_t hwcap,
                                                       uint64_t hwcap2);
/// Supply previous only for vector-state changes to preserve scalar type IDs.
RegisterLayout
CreateLinuxArm64RegisterLayout(const ArchSpec &arch,
                               const linux_arm64::RegisterConfig &config,
                               const RegisterDescription *previous = nullptr);
} // namespace lldb_private
#endif
