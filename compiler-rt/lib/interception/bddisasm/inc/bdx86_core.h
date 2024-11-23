/*
 * Copyright (c) 2020 Bitdefender
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef BDX86_CORE_H
#define BDX86_CORE_H

#include "bddisasm_types.h"
#include "bddisasm_status.h"
#include "bddisasm_version.h"

#include "bdx86_registers.h"
#include "bdx86_constants.h"
#include "bdx86_cpuidflags.h"


#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4214) // Bitfield in type other than int.
#pragma warning(disable: 4201) // Nonstandard extension used: nameless struct/union.
#endif

//
// Preferred vendor; the disassembler will try to be smart and disassemble as much as it can, but if there are
// encoding conflicts, than an alternate vendor can be selected. Note that this has effect only on conflicting
// encodings.
//
#define ND_VEND_ANY                 0   // Generic decode, include any vendor.
#define ND_VEND_INTEL               1   // Prefer Intel.
#define ND_VEND_AMD                 2   // Prefer AMD.
#define ND_VEND_MAX                 2

//
// These control what instructions should be decoded if they map onto the wide NOP space (0F 1A and 0F 1B). Those are
// tricky, because they might be NOP if the feature is disabled, but might be something else (even #UD) if the feature
// is enabled. Ergo, we allow the user to select whether said feature is on or off, so that he controls whether he
// sees the NOPs or the MPX/CET/CLDEMOTE/etc. instructions instead.
//
#define ND_FEAT_NONE                0x00    // No feature/mode enabled.
#define ND_FEAT_MPX                 0x01    // MPX support enabled.
#define ND_FEAT_CET                 0x02    // CET support enabled.
#define ND_FEAT_CLDEMOTE            0x04    // CLDEMOTE support enabled.
#define ND_FEAT_PITI                0x08    // PREFETCHITI support enabled.
#define ND_FEAT_APX                 0x10    // APX support enabled.
#define ND_FEAT_ALL                 0xFF    // Decode as if all features are enabled. This is default.

//
// Code type
//
#define ND_CODE_16                  0   // 16 bit decode mode.
#define ND_CODE_32                  1   // 32 bit decode mode.
#define ND_CODE_64                  2   // 64 bit decode mode.

//
// Data type
//
#define ND_DATA_16                  0   // 16 bit data size.
#define ND_DATA_32                  1   // 32 bit data size.
#define ND_DATA_64                  2   // 64 bit data size.

//
// Stack type
//
#define ND_STACK_16                 0   // 16 bit stack size.
#define ND_STACK_32                 1   // 32 bit stack size.
#define ND_STACK_64                 2   // 64 bit stack size.

//
// Addressing mode
//
#define ND_ADDR_16                  0   // 16 bit addressing.
#define ND_ADDR_32                  1   // 32 bit addressing.
#define ND_ADDR_64                  2   // 64 bit addressing.

//
// Operand mode/size
//
#define ND_OPSZ_16                  0   // 16 bit operand size.
#define ND_OPSZ_32                  1   // 32 bit operand size.
#define ND_OPSZ_64                  2   // 64 bit operand size.

//
// Vector mode/size
//
#define ND_VECM_128                 0   // 128 bit vector size.
#define ND_VECM_256                 1   // 256 bit vector size.
#define ND_VECM_512                 2   // 512 bit vector size.

//
// Encoding mode
//
#define ND_ENCM_LEGACY              0   // Legacy encoded instruction.
#define ND_ENCM_XOP                 1   // XOP encoded instruction.
#define ND_ENCM_VEX                 2   // VEX (bot 2B or 3B) encoded instruction.
#define ND_ENCM_EVEX                3   // EVEX encoded instruction.

//
// VEX mode
//
#define ND_VEXM_2B                  0   // 2B VEX prefix (0xC5).
#define ND_VEXM_3B                  1   // 3B VEX prefix (0xC4).

//
// EVEX mode
//
#define ND_EVEXM_EVEX               0   // Regular EVEX.
#define ND_EVEXM_VEX                1   // EVEX extension for VEX instructions.
#define ND_EVEXM_LEGACY             2   // EVEX extension for legacy instructions.
#define ND_EVEXM_COND               3   // EVEX extension for conditional instructions.


//
// Size definitions
//
#define ND_SIZE_8BIT                1           // 1 byte.
#define ND_SIZE_16BIT               2           // 1 word or 2 bytes.
#define ND_SIZE_32BIT               4           // 1 double word or 4 bytes.
#define ND_SIZE_48BIT               6           // 1 fword or 6 bytes.
#define ND_SIZE_64BIT               8           // 1 qword or 8 bytes.
#define ND_SIZE_80BIT               10          // 1 fpu word or 10 bytes.
#define ND_SIZE_112BIT              14          // FPU environment, 14 bytes.
#define ND_SIZE_128BIT              16          // 1 xmm word or 16 bytes.
#define ND_SIZE_224BIT              28          // FPU environment, 28 bytes.
#define ND_SIZE_256BIT              32          // 1 ymm word or 32 bytes.
#define ND_SIZE_384BIT              48          // 48 bytes, used for Key Locker handles.
#define ND_SIZE_512BIT              64          // 1 zmm word or 64 bytes. Used also for Key Locker handles.
#define ND_SIZE_752BIT              94          // FPU state, 94 bytes.
#define ND_SIZE_864BIT              108         // FPU state, 108 bytes.
#define ND_SIZE_4096BIT             512         // Extended state, 512 bytes.
#define ND_SIZE_1KB                 1024        // Tile register, 1KB.
#define ND_SIZE_CACHE_LINE          0xFFFFFFFE  // The size of a cache line.
#define ND_SIZE_UNKNOWN             0xFFFFFFFF  // Unknown/invalid size.


typedef ND_UINT32 ND_OPERAND_SIZE;

typedef ND_UINT32 ND_REG_SIZE;


//
// Prefix definitions
//
#define ND_PREFIX_G0_LOCK           0xF0    // LOCK prefix.
#define ND_PREFIX_G1_REPNE_REPNZ    0xF2    // REPNE/REPNZ prefix.
#define ND_PREFIX_G1_XACQUIRE       0xF2    // XACQUIRE prefix.
#define ND_PREFIX_G1_REPE_REPZ      0xF3    // REPE/REPZ prefix.
#define ND_PREFIX_G1_XRELEASE       0xF3    // XRELEASE prefix.
#define ND_PREFIX_G1_BND            0xF2    // BND prefix.
#define ND_PREFIX_G2_SEG_CS         0x2E    // CS segment override.
#define ND_PREFIX_G2_SEG_SS         0x36    // SS segment override.
#define ND_PREFIX_G2_SEG_DS         0x3E    // DS segment override.
#define ND_PREFIX_G2_SEG_ES         0x26    // ES segment override.
#define ND_PREFIX_G2_SEG_FS         0x64    // FS segment override.
#define ND_PREFIX_G2_SEG_GS         0x65    // GS segment override.
#define ND_PREFIX_G2_BR_NOT_TAKEN   0x2E    // Branch not taken hint.
#define ND_PREFIX_G2_BR_TAKEN       0x3E    // Branch taken hint.
#define ND_PREFIX_G2_BR_ALT         0x64    // Alternating branch hint.
#define ND_PREFIX_G2_NO_TRACK       0x3E    // Do not track prefix.
#define ND_PREFIX_G3_OPERAND_SIZE   0x66    // Operand size override.
#define ND_PREFIX_G4_ADDR_SIZE      0x67    // Address size override.

#define ND_PREFIX_REX_MIN           0x40    // First REX prefix.
#define ND_PREFIX_REX_MAX           0x4F    // Last REX prefix.
#define ND_PREFIX_VEX_2B            0xC5    // 2B VEX prefix.
#define ND_PREFIX_VEX_3B            0xC4    // 3B VEX prefix.
#define ND_PREFIX_XOP               0x8F    // XOP prefix.
#define ND_PREFIX_EVEX              0x62    // EVEX prefix.
#define ND_PREFIX_REX2              0xD5    // REX2 prefix.


//
// Instruction attributes
//
#define ND_FLAG_MODRM               0x0000000000000001  // The instruction has modrm.
#define ND_FLAG_F64                 0x0000000000000002  // The operand is forced to 64 bit. Size changing prefix 0x66 is ignored.
#define ND_FLAG_D64                 0x0000000000000004  // The default operand size is 64 bit. Size changing prefix 0x66
                                                        // changes the size to 16 bit. No 32 bit version can be encoded.
#define ND_FLAG_O64                 0x0000000000000008  // The instruction is available only in 64 bit mode.
#define ND_FLAG_I64                 0x0000000000000010  // The instruction is invalid in 64 bit mode.
#define ND_FLAG_COND                0x0000000000000020  // The instruction has a condition code encoded in low 4 bits of the opcode.
#define ND_FLAG_RESERVED0           0x0000000000000040  // Reserved.
#define ND_FLAG_VSIB                0x0000000000000080  // The instruction uses VSIB addressing mode.
#define ND_FLAG_MIB                 0x0000000000000100  // The instruction used MIB addressing mode.
#define ND_FLAG_LIG                 0x0000000000000200  // L flag inside XOP/VEX/EVEX is ignored.
#define ND_FLAG_WIG                 0x0000000000000400  // W flag inside XOP/VEX/EVEX is ignored.
#define ND_FLAG_3DNOW               0x0000000000000800  // The instruction is 3DNow!. The opcode is the last byte.
#define ND_FLAG_LOCKSP              0x0000000000001000  // MOV to/from CR in 16/32 bit, on AMD, extend the access to
                                                        // high 8 CRs via the LOCK prefix.
#define ND_FLAG_LOCK_SPECIAL        ND_FLAG_LOCKSP
#define ND_FLAG_MMASK               0x0000000000002000  // The instruction #UDs if K0 (no mask) is used.
#define ND_FLAG_NOMZ                0x0000000000004000  // The instruction #UDs if zeroing is used on memory.
#define ND_FLAG_NOL0                0x0000000000008000  // The instruction #UDs if vector length 128 is used.
#define ND_FLAG_NOA16               0x0000000000010000  // The instruction #UDs if 16-bit addressing is used.
#define ND_FLAG_MFR                 0x0000000000020000  // The Mod inside Mod R/M is forced to reg. No SIB/disp present.
#define ND_FLAG_VECTOR              0x0000000000040000  // The instruction is a SIMD instruction that operates on vector regs.
#define ND_FLAG_S66                 0x0000000000080000  // Special flag for mandatory 0x66 prefix that actually changes
                                                        // the default op length.
#define ND_FLAG_BITBASE             0x0000000000100000  // The instruction uses bitbase addressing mode.
#define ND_FLAG_AG                  0x0000000000200000  // The instruction is an address generator; no actual memory access.
#define ND_FLAG_SHS                 0x0000000000400000  // The instruction does a shadow stack access.
#define ND_FLAG_CETT                0x0000000000800000  // The instruction is CET tracked.
#define ND_FLAG_SERIAL              0x0000000001000000  // The instruction is serializing.
#define ND_FLAG_NORIPREL            0x0000000002000000  // The instruction #UDs if RIP-relative addressing is used.
#define ND_FLAG_NO_RIP_REL          ND_FLAG_NORIPREL
#define ND_FLAG_NO66                0x0000000004000000  // The instruction #UDs if 0x66 prefix is present.
#define ND_FLAG_SIBMEM              0x0000000008000000  // The instruction uses sibmem addressing (Intel AMX instructions).
#define ND_FLAG_I67                 0x0000000010000000  // Ignore the 0x67 prefix in 64 bit mode (Intel MPX instructions).
#define ND_FLAG_IER                 0x0000000020000000  // Ignore EVEX embedded rounding.
#define ND_FLAG_IWO64               0x0000000040000000  // Ignore VEX/EVEX.W outside 64 bit mode. It behaves as if it's 0.
#define ND_FLAG_NOREX2              0x0000000080000000  // The instruction #UDs if REX2 is present.
#define ND_FLAG_NOREP               0x0000000100000000  // The instruction #UDs if REP prefixes are present.
#define ND_FLAG_NO67                0x0000000200000000  // The instruction #UDs if 0x67 prefix is present.
#define ND_FLAG_NOV                 0x0000000400000000  // The instruction #UDs if XOP/VEX/EVEX.vvvv is not logical 0.
#define ND_FLAG_NOVP                0x0000000800000000  // The instruction #UDs if EVEX.v' is not logical 0.
#define ND_FLAG_SCALABLE            0x0000001000000000  // EVEX.pp can be 0 or 1, simulating the presence of 0x66 prefix.



//
// Accepted prefixes map
//
#define ND_PREF_REP                 0x0001      // The instruction supports REP prefix.
#define ND_PREF_REPC                0x0002      // The instruction supports REPZ/REPNZ prefixes.
#define ND_PREF_LOCK                0x0004      // The instruction supports LOCK prefix.
#define ND_PREF_HLE                 0x0008      // The instruction supports XACQUIRE/XRELEASE prefixes.
#define ND_PREF_XACQUIRE            0x0010      // The instruction supports only XACQUIRE.
#define ND_PREF_XRELEASE            0x0020      // The instruction supports only XRELEASE.
#define ND_PREF_BND                 0x0040      // The instruction supports BND prefix.
#define ND_PREF_BH                  0x0080      // The instruction supports branch hints.
#define ND_PREF_BHINT               ND_PREF_BH
#define ND_PREF_HLEWOL              0x0100      // HLE prefix is accepted without LOCK.
#define ND_PREF_HLE_WO_LOCK         ND_PREF_HLEWOL
#define ND_PREF_DNT                 0x0200      // The instruction supports the DNT (Do Not Track) CET prefix.


//
// Accepted decorators map. These are stored per-instruction. There are also per-operand indicators for
// each decorator, where applicable.
//
#define ND_DECO_ER                  0x01        // Embedded rounding is accepted.
#define ND_DECO_SAE                 0x02        // Suppress all Exceptions is accepted.
#define ND_DECO_ZERO                0x04        // Zeroing is accepted.
#define ND_DECO_MASK                0x08        // Masking is accepted.
#define ND_DECO_BROADCAST           0x10        // Memory broadcast is accepted.
#define ND_DECO_ND                  0x20        // New-data destination specifier accepted.
#define ND_DECO_NF                  0x40        // No-Flags specifier accepted.
#define ND_DECO_ZU                  0x80        // Zero-Upper semantic accepted.


//
// Operand access flags.
//
#define ND_ACCESS_NONE              0x00        // The operand is not accessed.
#define ND_ACCESS_READ              0x01        // The operand is read.
#define ND_ACCESS_WRITE             0x02        // The operand is written.
#define ND_ACCESS_COND_READ         0x04        // The operand is read only if some conditions are met.
#define ND_ACCESS_COND_WRITE        0x08        // The operand is written only if some conditions are met (ie: CMOVcc).
#define ND_ACCESS_ANY_READ          (ND_ACCESS_READ | ND_ACCESS_COND_READ)      // Any read mask.
#define ND_ACCESS_ANY_WRITE         (ND_ACCESS_WRITE | ND_ACCESS_COND_WRITE)    // Any write mask.
#define ND_ACCESS_PREFETCH          0x10        // The operand is prefetched.


//
// Condition definitions.
//
#define ND_COND_OVERFLOW            0x0         // OF
#define ND_COND_CARRY               0x2         // CF
#define ND_COND_BELOW               0x2         // CF
#define ND_COND_NOT_ABOVE_OR_EQUAL  0x2         // CF
#define ND_COND_ZERO                0x4         // ZF
#define ND_COND_EQUAL               0x4         // ZF
#define ND_COND_BELOW_OR_EQUAL      0x6         // CF | ZF
#define ND_COND_NOT_ABOVE           0x6         // CF | ZF
#define ND_COND_SIGN                0x8         // SF
#define ND_COND_PARITY              0xA         // PF
#define ND_COND_LESS                0xC         // SF ^ OF
#define ND_COND_LESS_OR_EQUAL       0xE         // (SF ^ OF) | ZF
#define ND_COND_NOT(p)              ((p) | 0x1) // Negates the predicate.


//
// Valid CPU modes.
//
// Group 1: ring
#define ND_MOD_R0                   0x00000001  // Instruction valid in ring 0.
#define ND_MOD_R1                   0x00000002  // Instruction valid in ring 1.
#define ND_MOD_R2                   0x00000004  // Instruction valid in ring 2.
#define ND_MOD_R3                   0x00000008  // Instruction valid in ring 3.

// Group 2: operating mode.
#define ND_MOD_REAL                 0x00000010  // Instruction valid in real mode.
#define ND_MOD_V8086                0x00000020  // Instruction valid in virtual 8086 mode.
#define ND_MOD_PROT                 0x00000040  // Instruction valid in protected mode.
#define ND_MOD_COMPAT               0x00000080  // Instruction valid in compatibility mode.
#define ND_MOD_LONG                 0x00000100  // Instruction valid in long mode.

// Group 3: misc modes.
#define ND_MOD_SMM                  0x00001000  // Instruction valid in System-Management Mode.
#define ND_MOD_SMM_OFF              0x00002000  // Instruction valid outside SMM.
#define ND_MOD_SGX                  0x00004000  // Instruction valid in SGX enclaves.
#define ND_MOD_SGX_OFF              0x00008000  // Instruction valid outside SGX enclaves.
#define ND_MOD_TSX                  0x00010000  // Instruction valid in TSX transactional regions.
#define ND_MOD_TSX_OFF              0x00020000  // Instruction valid outside TSX.


// Group 4: VMX
#define ND_MOD_VMXR                 0x00040000  // Instruction valid in VMX Root mode.
#define ND_MOD_VMXN                 0x00080000  // Instruction valid in VMX non-root mode.
#define ND_MOD_VMXR_SEAM            0x00100000  // Instruction valid in VMX root Secure Arbitration Mode.
#define ND_MOD_VMXN_SEAM            0x00200000  // Instruction valid in VMX non-root Secure Arbitration Mode.
#define ND_MOD_VMX_OFF              0x00400000  // Instruction valid outside VMX operation.

#define ND_MOD_RING_MASK            0x0000000F  // Valid ring mask.
#define ND_MOD_MODE_MASK            0x000001F0  // Valid mode mask.
#define ND_MOD_OTHER_MASK           0x0003F000  // Misc mask.
#define ND_MOD_VMX_MASK             0x007C0000  // VMX mask.

// For instructions valid in any operating mode.
#define ND_MOD_ANY                  0xFFFFFFFF  // Instruction valid in any mode.


//
// Misc constants
//
#define ND_MAX_INSTRUCTION_LENGTH   15          // 15 bytes is the maximum instruction length supported by the x86 arch.
#define ND_MAX_MNEMONIC_LENGTH      32          // Should do for now.
#define ND_MIN_BUF_SIZE             128         // Textual disassembly minimal buffer size.
#define ND_MAX_OPERAND              10          // No more than 10 operands/instruction, but I'm generous.
#define ND_MAX_REGISTER_SIZE        64          // Maximum register size - 64 bytes.
#define ND_MAX_GPR_REGS             32          // Max number of GPRs.
#define ND_MAX_SEG_REGS             8           // Max number of segment registers.
#define ND_MAX_FPU_REGS             8           // Max number of FPU registers.
#define ND_MAX_MMX_REGS             8           // Max number of MMX registers.
#define ND_MAX_SSE_REGS             32          // Max number of SSE registers.
#define ND_MAX_CR_REGS              32          // Max number of control registers.
#define ND_MAX_DR_REGS              32          // Max number of debug registers.
#define ND_MAX_TR_REGS              16          // Max number of test registers.
#define ND_MAX_MSK_REGS             8           // Max number of mask registers.
#define ND_MAX_BND_REGS             4           // Max number of bound registers.
#define ND_MAX_SYS_REGS             8           // Max number of system registers.
#define ND_MAX_X87_REGS             8           // Max number of x87 state/control registers registers.
#define ND_MAX_TILE_REGS            8           // Max number of tile registers.



//
// Misc macros.
//

// NOTE: Macros that accept a size (in bytes) are undefined if the size is not in the interval [1, 8].

// Sign extend 8 bit to 64 bit.
#define ND_SIGN_EX_8(x)             ND_SIGN_EX(1, x)
// Sign extend 16 bit to 64 bit.
#define ND_SIGN_EX_16(x)            ND_SIGN_EX(2, x)
// Sign extend 32 bit to 64 bit.
#define ND_SIGN_EX_32(x)            ND_SIGN_EX(4, x)
// Sign extend sz bytes to 64 bits.
#define ND_SIGN_EX(sz, x)           ((((ND_UINT64)(x)) & ND_SIZE_TO_MASK(sz)) | (~ND_SIZE_TO_MASK(sz) * ND_GET_SIGN(sz, x)))

// Trim 64 bits to sz bytes.
#define ND_TRIM(sz, x)              ((ND_UINT64)(x) & ND_SIZE_TO_MASK(sz))
// Returns most significant bit, given size in bytes sz.
#define ND_MSB(sz, x)               ((((x)) >> (((sz) * 8) - 1)) & 1)
// Returns least significant bit.
#define ND_LSB(sz, x)               ((x) & 1)
// Convert a size in bytes to a bitmask.
#define ND_SIZE_TO_MASK(sz)         (0xFFFFFFFFFFFFFFFFull >> ((8 - (sz)) * 8))
// Get bit at position bit from x.
#define ND_GET_BIT(bit, x)          (((x) >> (bit)) & 1)
// Return the sign of sz bytes long value x.
#define ND_GET_SIGN(sz, x)          ND_MSB(sz, x)
// Sets the sign of the sz bytes long value x.
#define ND_SET_SIGN(sz, x)          ND_SIGN_EX(sz, x)


#define ND_FETCH_64(b)              ((ND_UINT64)(((ND_UINT64)ND_FETCH_32((const ND_UINT8 *)(b))) | \
                                    (((ND_UINT64)ND_FETCH_32((const ND_UINT8 *)(b) + 4) << 32))))
#define ND_FETCH_32(b)              ((ND_UINT32)(((ND_UINT32)ND_FETCH_16((const ND_UINT8 *)(b))) | \
                                    (((ND_UINT32)ND_FETCH_16((const ND_UINT8 *)(b) + 2) << 16))))
#define ND_FETCH_16(b)              ((ND_UINT16)(((ND_UINT16)ND_FETCH_8 ((const ND_UINT8 *)(b))) | \
                                    (((ND_UINT16)ND_FETCH_8 ((const ND_UINT8 *)(b) + 1) << 8))))
#define ND_FETCH_8(b)               (*((const ND_UINT8 *)(b)))



//
// Helper macros which simply test the presence of various ND_FLAG_* in the instruction attributes.
//
#define ND_IS_3DNOW(ix)             (!!((ix)->Attributes & ND_FLAG_3DNOW))
#define ND_HAS_CONDITION(ix)        (!!((ix)->Attributes & ND_FLAG_COND))
#define ND_HAS_MODRM(ix)            (!!((ix)->Attributes & ND_FLAG_MODRM))
#define ND_HAS_VSIB(ix)             (!!((ix)->Attributes & ND_FLAG_VSIB))
#define ND_HAS_MIB(ix)              (!!((ix)->Attributes & ND_FLAG_MIB))
#define ND_HAS_VECTOR(ix)           (!!((ix)->Attributes & ND_FLAG_VECTOR))
#define ND_HAS_BITBASE(ix)          (!!((ix)->Attributes & ND_FLAG_BITBASE))
#define ND_HAS_AG(ix)               (!!((ix)->Attributes & ND_FLAG_AG))
#define ND_HAS_SIBMEM(ix)           (!!((ix)->Attributes & ND_FLAG_SIBMEM))
#define ND_HAS_SHS(ix)              (!!((ix)->Attributes & ND_FLAG_SHS))
#define ND_HAS_CETT(ix)             (!!((ix)->Attributes & ND_FLAG_CETT))

//
// Supported prefixes macros.
//
#define ND_REP_SUPPORT(ix)          (!!((ix)->ValidPrefixes.Rep))
#define ND_REPC_SUPPORT(ix)         (!!((ix)->ValidPrefixes.RepCond))
#define ND_LOCK_SUPPORT(ix)         (!!((ix)->ValidPrefixes.Lock))
#define ND_HLE_SUPPORT(ix)          (!!((ix)->ValidPrefixes.Hle))
#define ND_XACQUIRE_SUPPORT(ix)     (!!((ix)->ValidPrefixes.Xacquire))
#define ND_XRELEASE_SUPPORT(ix)     (!!((ix)->ValidPrefixes.Xrelease))
#define ND_BND_SUPPORT(ix)          (!!((ix)->ValidPrefixes.Bnd))
#define ND_BHINT_SUPPORT(ix)        (!!((ix)->ValidPrefixes.Bhint))
#define ND_DNT_SUPPORT(ix)          (!!((ix)->ValidPrefixes.Dnt))

//
// Decorators map macros.
//
#define ND_DECORATOR_SUPPORT(ix)    ((ix)->ValidDecorators.Raw != 0)
#define ND_MASK_SUPPORT(ix)         (!!((ix)->ValidDecorators.Mask))
#define ND_ZERO_SUPPORT(ix)         (!!((ix)->ValidDecorators.Zero))
#define ND_ER_SUPPORT(ix)           (!!((ix)->ValidDecorators.Er))
#define ND_SAE_SUPPORT(ix)          (!!((ix)->ValidDecorators.Sae))
#define ND_BROADCAST_SUPPORT(ix)    (!!((ix)->ValidDecorators.Broadcast))

// Generates a unique ID per register type, size and reg. The layout is the following:
//  - bits [63, 60] (4 bits)    - the operand type (ND_OP_REG)
//  - bits [59, 52] (8 bits)    - the register type
//  - bits [51, 36] (16 bits)   - the register size, in bytes
//  - bits [35, 30] (6 bits)    - the number of registers accessed starting with this reg (for block addressing)
//  - bits [29, 9] (21 bits)    - reserved
//  - bit 8                     - High8 indicator: indicates whether the reg is AH/CH/DH/BH
//  - bits [7, 0] (8 bits)      - the register ID
#define ND_OP_REG_ID(op)            (((ND_UINT64)((op)->Type & 0xF) << 60) |                                 \
                                     ((ND_UINT64)((op)->Info.Register.Type & 0xFF) << 52) |                  \
                                     ((ND_UINT64)((op)->Info.Register.Size & 0xFFFF) << 36) |                \
                                     ((ND_UINT64)((op)->Info.Register.Count & 0x3F) << 30) |                 \
                                     ((ND_UINT64)((op)->Info.Register.IsHigh8 & 0x1) << 8) |                 \
                                     ((ND_UINT64)((op)->Info.Register.Reg)))

// Example: ND_IS_OP_REG(op, ND_REG_GPR, 4, REG_ESP)
// Example: ND_IS_OP_REG(op, ND_REG_CR,  8, REG_CR3)
// Example: ND_IS_OP_REG(op, ND_REG_RIP, 8, 0)

// Checks if the indicated operand op is a register of type t, with size s and index r.
#define ND_IS_OP_REG(op, t, s, r)   (ND_OP_REG_ID(op) == (((ND_UINT64)(ND_OP_REG) << 60) |                   \
                                                          ((ND_UINT64)((t) & 0xFF) << 52) |                  \
                                                          ((ND_UINT64)((s) & 0xFFFF) << 36) |                \
                                                          ((ND_UINT64)(1) << 30) |                           \
                                                          ((ND_UINT64)(r))))

// Checks if the indicated operand op is a register of type t, with size s and index r.
#define ND_IS_OP_REG_EX(op, t, s, r, b, h)   (ND_OP_REG_ID(op) == (((ND_UINT64)(ND_OP_REG) << 60) |          \
                                                          ((ND_UINT64)((t) & 0xFF) << 52) |                  \
                                                          ((ND_UINT64)((s) & 0xFFFF) << 36) |                \
                                                          ((ND_UINT64)((b) & 0x3F) << 30) |                  \
                                                          ((ND_UINT64)((h) & 0x1) << 8) |                    \
                                                          ((ND_UINT64)(r))))

// Checjs if the indicated operand is the stack.
#define ND_IS_OP_STACK(op)          ((op)->Type == ND_OP_MEM && (op)->Info.Memory.IsStack)


//
// Operand types.
//
typedef enum _ND_OPERAND_TYPE
{
    ND_OP_NOT_PRESENT,      // Indicates the absence of any operand.
    ND_OP_REG,              // The operand is a register.
    ND_OP_MEM,              // The operand is located in memory.
    ND_OP_IMM,              // The operand is an immediate.
    ND_OP_OFFS,             // The operand is a relative offset.
    ND_OP_ADDR,             // The operand is an absolute far address, in the form seg:offset.
    ND_OP_ADDR_FAR = ND_OP_ADDR,
    ND_OP_ADDR_NEAR,        // The operand is an absolute near address, in the form target64.
    ND_OP_CONST,            // The operand is an implicit constant.
    ND_OP_BANK,             // An entire bank/set of registers are being accessed. Used in PUSHA/POPA/XSAVE/LOADALL.
    ND_OP_DFV,              // The operand is dfv (default flags value).
} ND_OPERAND_TYPE;


//
// Register types.
//
typedef enum _ND_REG_TYPE
{
    ND_REG_NOT_PRESENT,
    ND_REG_GPR,             // The register is a 8/16/32/64 bit general purpose register.
    ND_REG_SEG,             // The register is a segment register.
    ND_REG_FPU,             // The register is a 80-bit FPU register.
    ND_REG_MMX,             // The register is a 64-bit MMX register.
    ND_REG_SSE,             // The register is a 128/256/512 bit SSE vector register.
    ND_REG_CR,              // The register is a control register.
    ND_REG_DR,              // The register is a debug register.
    ND_REG_TR,              // The register is a test register.
    ND_REG_BND,             // The register is a bound register.
    ND_REG_MSK,             // The register is a mask register.
    ND_REG_TILE,            // The register is a tile register.
    ND_REG_MSR,             // The register is a model specific register.
    ND_REG_XCR,             // The register is a extended control register.
    ND_REG_SYS,             // The register is a system register.
    ND_REG_X87,             // The register is a x87 status/control register.
    ND_REG_MXCSR,           // The register is the MXCSR register.
    ND_REG_PKRU,            // The register is the PKRU register.
    ND_REG_SSP,             // The register is the SSP (Shadow Stack Pointer) register.
    ND_REG_FLG,             // The register is the FLAGS register.
    ND_REG_RIP,             // The register is the instruction pointer register.
    ND_REG_UIF,             // The register is the User Interrupt Flag.
} ND_REG_TYPE;


//
// Operand encoding types.
//
typedef enum _ND_OPERAND_ENCODING
{
    ND_OPE_NP,              // No encoding present.
    ND_OPE_R,               // Operand encoded in modrm.reg.
    ND_OPE_M,               // Operand encoded in modrm.rm.
    ND_OPE_V,               // Operand encoded in Xop/Vex/Evex/Mvex.(v')vvvv
    ND_OPE_D,               // Operand is encoded inside subsequent instruction bytes.
    ND_OPE_O,               // Operand is encoded in low 3 bit of the opcode.
    ND_OPE_I,               // Operand is an immediate.
    ND_OPE_C,               // Operand is CL.
    ND_OPE_1,               // Operand is 1.
    ND_OPE_L,               // Operand is reg encoded in immediate.
    ND_OPE_A,               // Operand is encoded in Evex.aaa.
    ND_OPE_E,               // Operand is a MSR or XCR encoded in ECX register.
    ND_OPE_S,               // Operand is implicit/suppressed. Not encoded anywhere.
} ND_OPERAND_ENCODING;


//
// Instruction tuple type; used to determine compressed displacement size for disp8 EVEX instructions. Note that
// most of the EVEX encoded instructions use the compressed displacement addressing scheme.
//
typedef enum _ND_TUPLE
{
    ND_TUPLE_None,
    ND_TUPLE_FV,            // Full Vector
    ND_TUPLE_HV,            // Half Vector
    ND_TUPLE_QV,            // Quarter Vector
    ND_TUPLE_T1S8,          // Tuple1 scalar, size 8 bit
    ND_TUPLE_T1S16,         // Tuple1 scalar, size 16 bit
    ND_TUPLE_T1S,           // Tuple1 scalar, size 32/64 bit
    ND_TUPLE_T1F,           // Tuple1 float, size 32/64 bit
    ND_TUPLE_T2,            // Tuple2, 64/128 bit
    ND_TUPLE_T4,            // Tuple4, 128/256 bit
    ND_TUPLE_T8,            // Tuple8, 256 bit
    ND_TUPLE_FVM,           // Full Vector Memory
    ND_TUPLE_HVM,           // Half Vector Memory
    ND_TUPLE_QVM,           // Quarter Vector Memory
    ND_TUPLE_OVM,           // Oct Vector Memory
    ND_TUPLE_M128,          // M128, 128 bit
    ND_TUPLE_DUP,           // DUP (VMOVDDUP)
    ND_TUPLE_T1_4X,         // 4 x 32 bit Memory Elements are referenced
} ND_TUPLE;


//
// EVEX rounding control.
//
typedef enum _ND_ROUNDING
{
    ND_RND_RNE,             // Round to nearest equal.
    ND_RND_RD,              // Round down.
    ND_RND_RU,              // Round up.
    ND_RND_RZ,              // round to zero.
} ND_ROUNDING;

//
// Exception types.
//
typedef enum _ND_EX_TYPE
{
    ND_EXT_None,

    // SSE/AVX exceptions.
    ND_EXT_1, 
    ND_EXT_2, 
    ND_EXT_3, 
    ND_EXT_4, 
    ND_EXT_5, 
    ND_EXT_6, 
    ND_EXT_7, 
    ND_EXT_8, 
    ND_EXT_9, 
    ND_EXT_10,
    ND_EXT_11,
    ND_EXT_12,
    ND_EXT_13,
    ND_EXT_14,

    // Opmask exceptios.
    ND_EXT_K20,
    ND_EXT_K21,

    // EVEX exceptions.
    ND_EXT_E1,
    ND_EXT_E1NF,
    ND_EXT_E2,
    ND_EXT_E3,
    ND_EXT_E3NF,
    ND_EXT_E4,
    ND_EXT_E4S,     // E4, with an additional case: if (dst == src1) or (dst == src2)
    ND_EXT_E4nb,
    ND_EXT_E4NF,
    ND_EXT_E4NFnb,
    ND_EXT_E5,
    ND_EXT_E5NF,
    ND_EXT_E6,
    ND_EXT_E6NF,
    ND_EXT_E7NM,
    ND_EXT_E9,
    ND_EXT_E9NF,
    ND_EXT_E10,
    ND_EXT_E10S,    // E10, with an additional case: if (dst == src1) or (dst == src2)
    ND_EXT_E10NF,
    ND_EXT_E11,
    ND_EXT_E12,
    ND_EXT_E12NP,

    // AMX exceptions.
    ND_EXT_AMX_E1,
    ND_EXT_AMX_E2,
    ND_EXT_AMX_E3,
    ND_EXT_AMX_E4,
    ND_EXT_AMX_E5,
    ND_EXT_AMX_E6,
    ND_EXT_AMX_E7,
    ND_EXT_AMX_E8,
    ND_EXT_AMX_E9,
    ND_EXT_AMX_E10,
    ND_EXT_AMX_E11,

    // AMX-EVEX exceptions.
    ND_EXT_AMX_EVEX_E1,
    ND_EXT_AMX_EVEX_E2,
    ND_EXT_AMX_EVEX_E3,
    ND_EXT_AMX_EVEX_E4,
    ND_EXT_AMX_EVEX_E5,
    ND_EXT_AMX_EVEX_E6,
    ND_EXT_AMX_EVEX_E7,
    ND_EXT_AMX_EVEX_E8,
        
    // APX-EVEX exceptions.
    ND_EXT_APX_EVEX_BMI,
    ND_EXT_APX_EVEX_CCMP,
    ND_EXT_APX_EVEX_WRSS,
    ND_EXT_APX_EVEX_WRUSS,
    ND_EXT_APX_EVEX_CFCMOV,
    ND_EXT_APX_EVEX_CMPCCXADD,
    ND_EXT_APX_EVEX_ENQCMD,
    ND_EXT_APX_EVEX_INT,
    ND_EXT_APX_EVEX_INVEPT,
    ND_EXT_APX_EVEX_INVPCID,
    ND_EXT_APX_EVEX_INVVPID,
    ND_EXT_APX_EVEX_KEYLOCKER,
    ND_EXT_APX_EVEX_KMOV,
    ND_EXT_APX_EVEX_PP2,
    ND_EXT_APX_EVEX_RAOINT,
    ND_EXT_APX_EVEX_SHA,
    ND_EXT_APX_EVEX_USER_MSR,
} ND_EX_TYPE;


//
// Operand access mode.
//
typedef union _ND_OPERAND_ACCESS
{
    ND_UINT8        Access;
    struct
    {
        ND_UINT8    Read : 1;       // The operand is read.
        ND_UINT8    Write : 1;      // The operand is written.
        ND_UINT8    CondRead : 1;   // The operand is read only under some conditions.
        ND_UINT8    CondWrite : 1;  // The operand is written only under some conditions.
        ND_UINT8    Prefetch : 1;   // The operand is prefetched.
    };
} ND_OPERAND_ACCESS;


//
// Operand flags.
//
typedef union _ND_OPERAND_FLAGS
{
    ND_UINT8        Flags;
    struct
    {
        ND_UINT8    IsDefault : 1;        // 1 if the operand is default. This also applies to implicit ops.
        ND_UINT8    SignExtendedOp1 : 1;  // 1 if the operand is sign extended to the first operands' size.
        ND_UINT8    SignExtendedDws : 1;  // 1 if the operand is sign extended to the default word size.
    };
} ND_OPERAND_FLAGS;


//
// Constant operand.
//
typedef struct _ND_OPDESC_CONSTANT
{
    ND_UINT64       Const;          // Instruction constant, ie ROL reg, 1.
} ND_OPDESC_CONSTANT;


//
// Immediate operand.
//
typedef struct _ND_OPDESC_IMMEDIATE
{
    ND_UINT64       Imm;            // Immediate. Only Size bytes are valid. The rest are undefined.
    ND_UINT8        RawSize;        // Raw size (how many bytes are encoded in the instruction).
} ND_OPDESC_IMMEDIATE;


//
// Relative offset operand.
//
typedef struct _ND_OPDESC_REL_OFFSET
{
    ND_UINT64       Rel;            // Relative offset (relative to the current RIP). Sign extended.
    ND_UINT8        RawSize;        // Raw size (how many bytes are encoded in the instruction).
} ND_OPDESC_RELOFFSET;


//
// Describes a register operand. Count registers are used starting with Reg.
//
typedef struct _ND_OPDESC_REGISTER
{
    ND_REG_TYPE     Type;           // The register type. See enum ND_REG_TYPE.
    ND_REG_SIZE     Size;           // Indicates the register size. This may not be equal to the Size
                                    // field, as a smaller amount of data may be processed from a
                                    // register (especially if we have a SSE register or a mask register).
                                    // Also note that as of now, 64 bytes is the maximum register size.
    ND_UINT32       Reg;            // The register number/ID.
    ND_UINT8        Count;          // The number of registers accessed, starting with Reg.

    ND_UINT8        IsHigh8:1;      // TRUE if this is AH, CH, DH or BH register.
    ND_UINT8        IsBlock:1;      // TRUE if this is a block register addressing.
    ND_UINT8        IsZeroUpper:1;  // TRUE if the upper register is zeroed.
} ND_OPDESC_REGISTER;


//
// Describes a seg:offset absolute effective address.
//
typedef struct _ND_OPDESC_ADDRESS
{
    // Size is the size of the address. Usually 4 (16 bit mode) or 6 (32 bit mode) or 10 (64 bit).
    ND_UINT64       Offset;     // Offset inside the segment.
    ND_UINT16       BaseSeg;    // Base segment selector of the address.
} ND_OPDESC_ADDRESS, ND_OPDESC_ADDRESS_FAR;


//
// Describes a 64-bit absolute effective address.
//
typedef struct _ND_OPDESC_ADDRESS_NEAR
{
    ND_UINT64       Target;     // Absolue 64-bit target address.
} ND_OPDESC_ADDRESS_NEAR;


//
// Shadow stack access types.
//
typedef enum _ND_SHSTK_ACCESS
{
    ND_SHSTK_NONE = 0,
    ND_SHSTK_EXPLICIT,      // Explicit memory operand accessed as shadow stack.
    ND_SHSTK_SSP_LD_ST,     // Shadow Stack Pointer (SSP) used as base for addressing using conventional load/store.
    ND_SHSTK_SSP_PUSH_POP,  // Shadow Stack Pointer (SSP) used as base for addressing using push/pop.
    ND_SHSTK_PL0_SSP,       // Privilege 0 SSP (IA32_PL0_SSP) used (SETSSBSY).
} ND_SHSTK_ACCESS;


//
// Describes a memory operand.
//
typedef struct _ND_OPDESC_MEMORY
{
    ND_BOOL         HasSeg:1;           // TRUE if segment is present & used.
    ND_BOOL         HasBase:1;          // TRUE if base register is present.
    ND_BOOL         HasIndex:1;         // TRUE if index & scale are present.
    ND_BOOL         HasDisp:1;          // TRUE if displacement is present.
    ND_BOOL         HasCompDisp:1;      // TRUE if compressed disp8 is used (EVEX instructions).
    ND_BOOL         HasBroadcast:1;     // TRUE if the memory operand is a broadcast operand.

    ND_BOOL         IsRipRel:1;         // TRUE if this is a rip-relative addressing. Base, Index, Scale are
                                        // all ignored.
    ND_BOOL         IsStack:1;          // TRUE if this is a stack op. Note that explicit stack accesses are not
                                        // included (eg: mov eax, [rsp] will NOT set IsStack).
    ND_BOOL         IsString:1;         // TRUE for [RSI] and [RDI] operands inside string operations.
    ND_BOOL         IsShadowStack:1;    // TRUE if this is a shadow stack access. Check out ShStkType for more info.
    ND_BOOL         IsDirect:1;         // TRUE if direct addressing (MOV [...], EAX, 0xA3).
    ND_BOOL         IsBitbase:1;        // TRUE if this is a bit base. Used for BT* instructions. The bitbase
                                        // stored in the second operand must be added to the linear address.
    ND_BOOL         IsAG:1;             // TRUE if the memory operand is address generation and no mem access is
                                        // made.
    ND_BOOL         IsMib:1;            // TRUE if MIB addressing is used (MPX instructions).
    ND_BOOL         IsVsib:1;           // TRUE if the index register selects a vector register.
    ND_BOOL         IsSibMem:1;         // TRUE if the addressing uses sibmem (AMX instructions).

    ND_UINT8        BaseSize;           // Base size, in bytes. Max 8 bytes.
    ND_UINT8        IndexSize;          // Ditto for index size. Max 64 bytes for VSIB.
    ND_UINT8        DispSize;           // Displacement size. Max 4 bytes.
    ND_UINT8        CompDispSize;       // Compressed displacement size - 1, 2, 4, 8, 16, 32, 64.

    ND_UINT8        ShStkType;          // Shadow stack access type. Check out ND_SHSTK_ACCESS.

    union
    {
        struct
        {
            ND_UINT8    IndexSize;      // VSIB index size.
            ND_UINT8    ElemSize;       // VSIB element size.
            ND_UINT8    ElemCount;      // Number of elements scattered/gathered/prefetched.
        } Vsib;                         // Valid if HasVsib is set.

        struct
        {
            ND_UINT8    Count;          // Number of times to broadcast the element.
            ND_UINT8    Size;           // Size of one element.
        } Broadcast;                    // Valid if HasBroadcast is set.
    };

    ND_UINT8        Seg:3;              // Base segment used to address the memory. 0 = es, 1 = cs, etc.
    ND_UINT8        Base:5;             // Base register. Can only be a GPR.
    ND_UINT8        Index;              // Index register. Can be a vector reg (ZMM0-ZMM31).
    ND_UINT8        Scale:4;            // Scale: 1, 2, 4 or 8. Always present if Index is present.

    ND_UINT64       Disp;               // Sign extended displacement.

} ND_OPDESC_MEMORY;


//
// Describes a Default Flags Value operand.
//
typedef struct _ND_OPDESC_DEFAULT_FLAGS
{
    ND_UINT8        CF:1;
    ND_UINT8        ZF:1;
    ND_UINT8        SF:1;
    ND_UINT8        OF:1;
} ND_OPDESC_DEFAULT_FLAGS;


//
// Describes a decorator that applies to an operand.
//
typedef struct _ND_OPERAND_DECORATOR
{
    ND_BOOL         HasMask:1;      // TRUE if mask is present, 0 otherwise.
    ND_BOOL         HasZero:1;      // TRUE if zeroing will be made, 0 if merging will be made.
    ND_BOOL         HasBroadcast:1; // TRUE if broadcasting is being made. Valid only for memory operands.

    // Mask register specifier.
    ND_UINT8        Msk:3;          // Mask register used. Only k0-k7 can be encoded.

} ND_OPERAND_DECORATOR;


//
// Extended operand information.
//
typedef struct _ND_OPERAND
{
    ND_UINT8                    Type:4;         // Operand type. One of ND_OPERAND_TYPE enumerations.
    ND_UINT8                    Encoding:4;     // Where is the operand encoded. One of ND_OPERAND_ENCODING enumerations.
    ND_OPERAND_ACCESS           Access;         // Access mode (read, write, read-write, etc.)
    ND_OPERAND_FLAGS            Flags;          // Misc operand flags.
    ND_OPERAND_DECORATOR        Decorator;      // Decorator information.
    ND_OPERAND_SIZE             Size;           // Operand size in bytes. This should be used when operating with
                                                // the operand. It includes sign-extension or zero-extensions.
                                                // Note that the Size field indicates the actual amount of data
                                                // used for processing. If the operand type is a register, it MAY NOT
                                                // indicate the register size. Use the Info.Register.Size
                                                // field to get the actual register size.

    // Depending in the Type field, one of these subfields contains information about the operand.
    // Althoug Immediate, RelativeOffset & Address are all immediate payloads, they are kept separate, due
    // the slightly different semantic:
    // 1. Immediate is a plain immediate quantity
    // 2. RelativeOffset is an immediate added to the instruction pointer
    // 3. Address is an mmediate formed formed of a segment:offset pair
    union
    {
        ND_OPDESC_REGISTER      Register;       // Register, if operand type if ND_OP_REG.
        ND_OPDESC_MEMORY        Memory;         // Memory, if operand type is ND_OP_MEM.
        ND_OPDESC_IMMEDIATE     Immediate;      // Immediate, if operand type is ND_OP_IMM.
        ND_OPDESC_RELOFFSET     RelativeOffset; // Relative offset, if operand type is ND_OP_OFFS.
        ND_OPDESC_ADDRESS_FAR   Address;        // Address, seg:offset form, if operand type is ND_OP_ADDR.
        ND_OPDESC_ADDRESS_NEAR  AddressNear;    // Address, target64 form, if operand type is ND_OP_ADDR_NEAR.
        ND_OPDESC_CONSTANT      Constant;       // Constant, if operand type is ND_OP_CONST.
        ND_OPDESC_DEFAULT_FLAGS DefaultFlags;   // Default flags value, if operand type is ND_OP_DFV.
    } Info;

} ND_OPERAND, *PND_OPERAND;


//
// REX prefix.
//
typedef union _ND_REX
{
    ND_UINT8        Rex;
    struct
    {
        ND_UINT8    b : 1;      // b (rm or low opcode) extension field.
        ND_UINT8    x : 1;      // x (index) extension field.
        ND_UINT8    r : 1;      // r (reg) extension field.
        ND_UINT8    w : 1;      // w (size) extension field. Promotes to 64 bit.
    };
} ND_REX;

//
// REX2 prefix.
//
typedef union _ND_REX2
{
    ND_UINT8        Rex2[2];
    struct
    {
        ND_UINT8    op;         // 0xD5

        ND_UINT8    b3 : 1;     // B3 (rm or low opcode) extension field.
        ND_UINT8    x3 : 1;     // X3 (index) extension field.
        ND_UINT8    r3 : 1;     // R3 (reg) extension field.
        ND_UINT8    w  : 1;     // W (size) extension field. Promotes to 64 bit.
        ND_UINT8    b4 : 1;     // B4 (rm or low opcode) extension field.
        ND_UINT8    x4 : 1;     // X4 (index) extension field.
        ND_UINT8    r4 : 1;     // R4 (reg) extension field.
        ND_UINT8    m0 : 1;     // M0 map ID.
    };
} ND_REX2;

//
// Mod R/M byte.
//
typedef union _ND_MODRM
{
    ND_UINT8        ModRm;
    struct
    {
        ND_UINT8    rm : 3;     // rm field.
        ND_UINT8    reg : 3;    // reg field.
        ND_UINT8    mod : 2;    // mod field. Indicates memory access (0, 1 or 2), or register access (3).
    };
} ND_MODRM;

//
// SIB byte.
//
typedef union _ND_SIB
{
    ND_UINT8        Sib;
    struct
    {
        ND_UINT8    base : 3;   // Base register.
        ND_UINT8    index : 3;  // Index register.
        ND_UINT8    scale : 2;  // Scale.
    };
} ND_SIB;

//
// 2-bytes VEX prefix.
//
typedef union _ND_VEX2
{
    ND_UINT8        Vex[2];
    struct
    {
        ND_UINT8    op;          // 0xC5

        ND_UINT8    p : 2;       // p0, p1
        ND_UINT8    l : 1;       // L
        ND_UINT8    v : 4;       // ~v0, ~v1, ~v2, ~v3
        ND_UINT8    r : 1;       // ~R
    };
} ND_VEX2;

//
// 3-bytes VEX prefix.
//
typedef union _ND_VEX3
{
    ND_UINT8        Vex[3];
    struct
    {
        ND_UINT8    op;         // 0xC4

        ND_UINT8    m : 5;      // m0, m1, m2, m3, m4
        ND_UINT8    b : 1;      // ~B
        ND_UINT8    x : 1;      // ~X
        ND_UINT8    r : 1;      // ~R

        ND_UINT8    p : 2;      // p0, p1
        ND_UINT8    l : 1;      // L
        ND_UINT8    v : 4;      // ~v0, ~v1, ~v2, ~v3
        ND_UINT8    w : 1;      // W
    };
} ND_VEX3;

//
// XOP prefix.
//
typedef union _ND_XOP
{
    ND_UINT8        Xop[3];
    struct
    {
        ND_UINT8    op;         // 0x8F

        ND_UINT8    m : 5;      // m0, m1, m2, m3, m4
        ND_UINT8    b : 1;      // ~B
        ND_UINT8    x : 1;      // ~X
        ND_UINT8    r : 1;      // ~R

        ND_UINT8    p : 2;      // p0, p1
        ND_UINT8    l : 1;      // L
        ND_UINT8    v : 4;      // ~v0, ~v1, ~v2, ~v3
        ND_UINT8    w : 1;      // W
    };
} ND_XOP;

//
// EVEX prefix.
//
typedef union _ND_EVEX
{
    ND_UINT8        Evex[4];
    struct
    {
        ND_UINT8    op;         // 0x62

        ND_UINT8    m : 3;      // m0, m1, m2. Indicates opcode map.
        ND_UINT8    b4 : 1;     // B4 (repurposed from a hard-coded 0 bit).
        ND_UINT8    rp : 1;     // ~R' or ~R4
        ND_UINT8    b : 1;      // ~B or ~B3
        ND_UINT8    x : 1;      // ~X or ~X3
        ND_UINT8    r : 1;      // ~R or ~R3

        ND_UINT8    p : 2;      // p0, p1
        ND_UINT8    u : 1;      // ~U (repurposed from a hard-coded 1 bit).
        ND_UINT8    v : 4;      // ~v0, ~v1, ~v2, ~v3
        ND_UINT8    w : 1;      // W

        ND_UINT8    a : 3;      // a0, a1, a2
        ND_UINT8    vp : 1;     // ~V'
        ND_UINT8    bm : 1;     // b
        ND_UINT8    l : 2;      // L'L
        ND_UINT8    z : 1;      // z
    };
} ND_EVEX;


//
// Describes the CPUID leaf, sub-leaf, register & bit that indicate whether an instruction is supported or not.
// If Leaf == ND_CFF_NO_LEAF, the instruction is supported on any CPU, and no CPUID flag exists.
// If SubLeaf == ND_CFF_NO_SUBLEAF, there is no subleaf to check.
//
typedef union _ND_CPUID_FLAG
{
    ND_UINT64       Flag;
    struct
    {
        ND_UINT32   Leaf;          // CPUID leaf. ND_CFF_NO_LEAF if not applicable.
        ND_UINT32   SubLeaf : 24;  // CPUID sub-leaf. ND_CFF_NO_SUBLEAF if not applicable.
        ND_UINT32   Reg : 3;       // The register that contains info regarding the instruction.
        ND_UINT32   Bit : 5;       // Bit inside the register that indicates whether the instruction is present.
    };
} ND_CPUID_FLAG;


//
// Each instruction may accept one or more prefixes. This structure indicates which prefixes are valid for the 
// given instruction.
//
typedef union _ND_VALID_PREFIXES
{
    ND_UINT16       Raw;
    struct
    {
        ND_UINT16   Rep : 1;        // The instruction supports REP prefix.
        ND_UINT16   RepCond : 1;    // The instruction supports REPZ/REPNZ prefixes.
        ND_UINT16   Lock : 1;       // The instruction supports LOCK prefix.
        ND_UINT16   Hle : 1;        // The instruction supports XACQUIRE/XRELEASE prefixes.
        ND_UINT16   Xacquire : 1;   // The instruction supports only XACQUIRE.
        ND_UINT16   Xrelease : 1;   // The instruction supports only XRELEASE.
        ND_UINT16   Bnd : 1;        // The instruction supports BND prefix.
        ND_UINT16   Bhint : 1;      // The instruction supports branch hints.
        ND_UINT16   HleNoLock : 1;  // HLE prefix is accepted without LOCK.
        ND_UINT16   Dnt : 1;        // The instruction supports the DNT (Do Not Track) CET prefix.
    };
} ND_VALID_PREFIXES, *PND_VALID_PREFIXES;


//
// Each instruction may accept several decorators. This instruction indicates which decorators are valid for the 
// given instruction.
//
typedef union _ND_VALID_DECORATORS
{
    ND_UINT8        Raw;
    struct
    {
        ND_UINT8    Er : 1;         // The instruction supports embedded rounding mode.
        ND_UINT8    Sae : 1;        // The instruction supports suppress all exceptions mode.
        ND_UINT8    Zero : 1;       // The instruction supports zeroing.
        ND_UINT8    Mask : 1;       // The instruction supports mask registers.
        ND_UINT8    Broadcast : 1;  // The instruction supports broadcast.
        ND_UINT8    Nd : 1;         // The instruction supports new data destination.
        ND_UINT8    Nf : 1;         // The instruction supports no-flags update.
        ND_UINT8    Zu : 1;         // The instruction supports zero-upper semantic.
    };
} ND_VALID_DECORATORS, *PND_VALID_DECORATORS;


//
// Each instruction is valid or invalid in any certain mode. This indicates which modes the instruction is valid in.
// If the bit is set, the isntruction is valid in that given mode.
//
typedef union _ND_VALID_MODES
{
    ND_UINT32       Raw;
    struct
    {
        // Group 1: privilege level.
        ND_UINT32   Ring0 : 1;     // The instruction is valid in ring 0.
        ND_UINT32   Ring1 : 1;     // The instruction is valid in ring 1.
        ND_UINT32   Ring2 : 1;     // The instruction is valid in ring 2.
        ND_UINT32   Ring3 : 1;     // The instruction is valid in ring 3.

        // Group 2: operating mode - the CPU can be on only one of these modes at any moment.
        ND_UINT32   Real : 1;      // The instruction is valid in real mode.
        ND_UINT32   V8086 : 1;     // The instruction is valid in Virtual 8086 mode.
        ND_UINT32   Protected : 1; // The instruction is valid in protected mode (32 bit).
        ND_UINT32   Compat : 1;    // The instruction is valid in compatibility mode (32 bit in 64 bit).
        ND_UINT32   Long : 1;      // The instruction is valid in long mode.

        ND_UINT32   Reserved : 3;  // Reserved for padding/future use.

        // Group 3: special modes - these may be active inside other modes (example: TSX in Long mode).
        ND_UINT32   Smm : 1;       // The instruction is valid in System Management Mode.
        ND_UINT32   SmmOff : 1;    // The instruction is valid outside SMM.
        ND_UINT32   Sgx : 1;       // The instruction is valid in SGX mode.
        ND_UINT32   SgxOff : 1;    // The instruction is valid outside SGX.
        ND_UINT32   Tsx : 1;       // The instruction is valid in transactional regions.
        ND_UINT32   TsxOff : 1;    // The instruction is valid outside TSX.

        // Group 4: VMX mode - they engulf all the other modes.
        ND_UINT32   VmxRoot : 1;   // The instruction is valid in VMX root mode.
        ND_UINT32   VmxNonRoot : 1;// The instruction is valid in VMX non root mode.
        ND_UINT32   VmxRootSeam : 1;   // The instruction is valid in VMX root SEAM.
        ND_UINT32   VmxNonRootSeam : 1;// The instruction is valid in VMX non-root SEAM.
        ND_UINT32   VmxOff : 1;    // The instruction is valid outside VMX operation.
        
    };
} ND_VALID_MODES, *PND_VALID_MODES;


//
// RFLAGS register. This structure reflects the actual position of each flag insdide the RFLAGS register, so it can
// be used for direct processing.
//
typedef union _ND_RFLAGS
{
    ND_UINT32       Raw;
    struct
    {
        ND_UINT32   CF : 1;         // Carry flag.
        ND_UINT32   Reserved1 : 1;  // Reserved, must be 1.
        ND_UINT32   PF : 1;         // Parity flag.
        ND_UINT32   Reserved2 : 1;  // Reserved.
        ND_UINT32   AF : 1;         // Auxiliary flag.
        ND_UINT32   Reserved3 : 1;  // Reserved.
        ND_UINT32   ZF : 1;         // Zero flag.
        ND_UINT32   SF : 1;         // Sign flag.
        ND_UINT32   TF : 1;         // Trap flag.
        ND_UINT32   IF : 1;         // Interrupt flag.
        ND_UINT32   DF : 1;         // Direction flag.
        ND_UINT32   OF : 1;         // Overflow flag.
        ND_UINT32   IOPL : 2;       // I/O privilege level flag.
        ND_UINT32   NT : 1;         // Nested task flag.
        ND_UINT32   Reserved4 : 1;  // Reserved.
        ND_UINT32   RF : 1;         // Resume flag.
        ND_UINT32   VM : 1;         // Virtual mode flag.
        ND_UINT32   AC : 1;         // Alignment check flag.
        ND_UINT32   VIF : 1;        // Virtual interrupts flag.
        ND_UINT32   VIP : 1;        // Virtual interrupt pending flag.
        ND_UINT32   ID : 1;         // CPUID identification flag.
    };
} ND_RFLAGS, *PND_RFLAGS;


#define ND_FPU_FLAG_SET_0           0   // The FPU status flag is cleared to 0.
#define ND_FPU_FLAG_SET_1           1   // The FPU status flag is set to 1.
#define ND_FPU_FLAG_MODIFIED        2   // The FPU status flag is modified according to a result.
#define ND_FPU_FLAG_UNDEFINED       3   // The FPU status flag is undefined or unaffected.

//
// FPU status flags. Each status flag can be one of ND_FPU_FLAG*.
//
typedef union _ND_FPU_FLAGS
{
    ND_UINT8        Raw;
    struct
    {
        ND_UINT8    C0 : 2;         // C0 flag access mode. See ND_FPU_FLAG_*.
        ND_UINT8    C1 : 2;         // C1 flag access mode. See ND_FPU_FLAG_*.
        ND_UINT8    C2 : 2;         // C2 flag access mode. See ND_FPU_FLAG_*.
        ND_UINT8    C3 : 2;         // C3 flag access mode. See ND_FPU_FLAG_*.
    };
} ND_FPU_FLAGS, *PND_FPU_FLAGS;


#define ND_SIMD_EXC_IE              0x01 // Invalid Operation Exception.
#define ND_SIMD_EXC_DE              0x02 // Denormal Exception.
#define ND_SIMD_EXC_ZE              0x04 // Divide-by-Zero Exception.
#define ND_SIMD_EXC_OE              0x08 // Overflow Exception.
#define ND_SIMD_EXC_UE              0x10 // Underflow Exception.
#define ND_SIMD_EXC_PE              0x20 // Precision Exception.

//
// SIMD Floating-Point Exceptions. These values are the same as lower 6 bits in MXCSR. The Raw field
// is a combination of ND_SIMD_EXC_* values, and is the same as the invidiual bitfields.
//
typedef union _ND_SIMD_EXCEPTIONS
{
    ND_UINT8        Raw;
    struct
    {
        ND_UINT8    IE : 1;         // Invalid Operation Exception.
        ND_UINT8    DE : 1;         // Denormal Exception.
        ND_UINT8    ZE : 1;         // Divide-by-Zero Exception.
        ND_UINT8    OE : 1;         // Overflow Exception.
        ND_UINT8    UE : 1;         // Underflow Exception.
        ND_UINT8    PE : 1;         // Precision Exception.
    };
} ND_SIMD_EXCEPTIONS;


//
// Branch information.
//
typedef struct _ND_BRANCH_INFO
{
    ND_UINT8        IsBranch : 1;
    ND_UINT8        IsConditional : 1;
    ND_UINT8        IsIndirect : 1;
    ND_UINT8        IsFar : 1;
} ND_BRANCH_INFO;



//
// Describes a decoded instruction. All the possible information about the instruction is contained in this structure.
// You don't have to call any other APIs to gather any more info about it.
//
typedef struct _INSTRUX
{
    ND_UINT8            DefCode:2;                  // ND_CODE_*. Indicates disassembly mode.
    ND_UINT8            DefData:2;                  // ND_DATA_*. Indicates default data size.
    ND_UINT8            DefStack:2;                 // ND_STACK_*. Indicates default stack pointer width.
    ND_UINT8            AddrMode:2;                 // ND_ADDR_*. Indicates addressing mode.
    ND_UINT8            OpMode:2;                   // ND_OPSZ_*. Indicates operand mode/size.
    ND_UINT8            EfOpMode:2;                 // ND_OPSZ_*. Indicates effective operand mode/size.
    ND_UINT8            VecMode:2;                  // ND_VECM_*. Indicates vector length. Valid only for vector instructions.
    ND_UINT8            EfVecMode:2;                // ND_VECM_*. Indicates effective vector length. Valid only for vector instructions.
    ND_UINT8            EncMode:4;                  // ND_ENCM_*. Indicates encoding mode.
    ND_UINT8            VexMode:2;                  // ND_VEX_*.  Indicates the VEX mode, if any. Valid only if HasVex set.
    ND_UINT8            EvexMode:4;                 // ND_EVEX_*. Indicates EVEX extension, if any. Valid only if HasEvex set.
    ND_UINT8            VendMode:4;                 // ND_VEND_*. Indicates vendor mode.
    ND_UINT8            FeatMode;                   // ND_FEAT_*. Indicates which features are enabled.
    
    // Present prefixes. Note that even if a prefix is marked as being present in the encoding, it does not necessary mean
    // that the prefix is actually used. Check Is*Enabled fields to check if the prefix is enabled & used. In some cases,
    // prefixes are ignored, even if present.
    ND_BOOL             HasRex:1;                   // TRUE - REX is present.
    ND_BOOL             HasRex2:1;                  // TRUE - REX2 is present.
    ND_BOOL             HasVex:1;                   // TRUE - VEX is present.
    ND_BOOL             HasXop:1;                   // TRUE - XOP is present.
    ND_BOOL             HasEvex:1;                  // TRUE - EVEX is present.
    ND_BOOL             HasOpSize:1;                // TRUE - 0x66 present.
    ND_BOOL             HasAddrSize:1;              // TRUE - 0x67 present.
    ND_BOOL             HasLock:1;                  // TRUE - 0xF0 present.
    ND_BOOL             HasRepnzXacquireBnd:1;      // TRUE - 0xF2 present.
    ND_BOOL             HasRepRepzXrelease:1;       // TRUE - 0xF3 present.
    ND_BOOL             HasSeg:1;                   // TRUE - segment override present.

    // Present encoding components.
    ND_BOOL             HasModRm:1;                 // TRUE - we have valid MODRM.
    ND_BOOL             HasSib:1;                   // TRUE - we have valid SIB.
    ND_BOOL             HasDisp:1;                  // TRUE - the instruction has displacement.
    ND_BOOL             HasAddr:1;                  // TRUE - the instruction contains a direct far address (ie, CALL far 0x9A)
    ND_BOOL             HasAddrNear:1;              // TRUE - the instruction contains a direct near address (ie, CALL far 0x9A)
    ND_BOOL             HasMoffset:1;               // TRUE - the instruction contains a moffset (ie, MOV al, [mem], 0xA0)
    ND_BOOL             HasRelOffs:1;               // TRUE - the instruction contains a relative offset (ie, Jcc 0x7x).
    ND_BOOL             HasImm1:1;                  // TRUE - immediate present.
    ND_BOOL             HasImm2:1;                  // TRUE - second immediate present.
    ND_BOOL             HasSseImm:1;                // TRUE - SSE immediate that encodes additional registers is present.

    // Present decorators & EVEX info.
    ND_BOOL             HasCompDisp:1;              // TRUE - the instruction uses compressed displacement.
    ND_BOOL             HasBroadcast:1;             // TRUE - the instruction uses broadcast addressing.
    ND_BOOL             HasMask:1;                  // TRUE - the instruction has mask.
    ND_BOOL             HasZero:1;                  // TRUE - the instruction uses zeroing.
    ND_BOOL             HasEr:1;                    // TRUE - the instruction has embedded rounding.
    ND_BOOL             HasSae:1;                   // TRUE - the instruction has SAE.
    ND_BOOL             HasNd:1;                    // TRUE - the instruction uses New-Data Destination.
    ND_BOOL             HasNf:1;                    // TRUE - the instruction uses NoFlags update.
    ND_BOOL             HasZu:1;                    // TRUE - the instruction has ZeroUpper.
    ND_BOOL             HasIgnEr:1;                 // TRUE - the instruction ignores embedded rounding.

    // Mandatory prefixes.
    ND_BOOL             HasMandatory66:1;           // 0x66 is mandatory prefix. Does not behave as size-changing prefix.
    ND_BOOL             HasMandatoryF2:1;           // 0x66 is mandatory prefix. Does not behave as REP prefix.
    ND_BOOL             HasMandatoryF3:1;           // 0x66 is mandatory prefix. Does not behave as REP prefix.

    // Prefix activation. Use these fields to check whether a prefix is present & active for the instruction.
    ND_BOOL             IsLockEnabled:1;            // TRUE - LOCK is present & used.
    ND_BOOL             IsRepEnabled:1;             // TRUE - REP is present & used.
    ND_BOOL             IsRepcEnabled:1;            // TRUE - REPZ/REPNZ is present & used.
    ND_BOOL             IsXacquireEnabled:1;        // TRUE - the instruction is XACQUIRE enabled.
    ND_BOOL             IsXreleaseEnabled:1;        // TRUE - the instruction is XRELEASE enabled.
    ND_BOOL             IsBhintEnabled:1;           // TRUE - branch hints valid & used.
    ND_BOOL             IsBndEnabled:1;             // TRUE - BND prefix valid & used.
    ND_BOOL             IsDntEnabled:1;             // TRUE - DNT prefix valid & used.
    ND_BOOL             IsRepeated:1;               // TRUE - the instruction is REPed up to RCX times.
    ND_BOOL             IsCetTracked:1;             // TRUE - this is an indirect CALL/JMP that is CET tracked.

    // Misc.
    ND_UINT8            IsRipRelative:1;            // TRUE - the instruction uses RIP relative addressing.
    ND_UINT8            RoundingMode:2;             // EVEX rounding mode, if present. One of ND_ROUNDING.

    // Instruction components lengths. Will be 0 if the given field is not present.
    ND_UINT8            Length;                     // 1-15 valid. Instructions longer than 15 bytes will cause #GP.
    ND_UINT8            WordLength:4;               // The length of the instruction word. 2, 4 or 8.
    ND_UINT8            StackWords:4;               // Number of words accessed on/from the stack. 0-15.

    ND_UINT8            PrefLength:4;               // The total number of bytes consumed by prefixes. This will also be 
                                                    // the offset to the first opcode. The primary opcode will always be 
                                                    // the last one, so at offset PrefixesLength + OpcodeLength - 1
    ND_UINT8            OpLength:4;                 // Number of opcode bytes. Max 3.
    ND_UINT8            DispLength:4;               // Displacement length, in bytes. Maximum 4.
    ND_UINT8            AddrLength:4;               // Absolute address length, in bytes. Maximum 8 bytes.
    ND_UINT8            MoffsetLength:4;            // Memory offset length, in bytes. Maximum 8 bytes.
    ND_UINT8            Imm1Length:4;               // First immediate length, in bytes. Maximum 8 bytes.
    ND_UINT8            Imm2Length:4;               // Second immediate length, in bytes. Maximum 1 byte.
    ND_UINT8            RelOffsLength:4;            // Relative offset length, in bytes. Maximum 4 bytes.

    // Instruction components offsets. Will be 0 if the given field is not present. Prefixes ALWAYS start at offset 0.
    ND_UINT8            OpOffset:4;                 // The offset of the first opcode, inside the instruction.
    ND_UINT8            MainOpOffset:4;             // The offset of the nominal opcode, inside the instruction.
    ND_UINT8            DispOffset:4;               // The offset of the displacement, inside the instruction
    ND_UINT8            AddrOffset:4;               // The offset of the hard-coded address.
    ND_UINT8            MoffsetOffset:4;            // The offset of the absolute address, inside the instruction
    ND_UINT8            Imm1Offset:4;               // The offset of the immediate, inside the instruction
    ND_UINT8            Imm2Offset:4;               // The offset of the second immediate, if any, inside the instruction
    ND_UINT8            RelOffsOffset:4;            // The offset of the relative offset used in instruction.
    ND_UINT8            SseImmOffset:4;             // The offset of the SSE immediate, if any, inside the instruction.
    ND_UINT8            ModRmOffset:4;              // The offset of the mod rm byte inside the instruction, if any.
                                                    // If SIB is also present, it will always be at ModRmOffset + 1.

    // This structures contains the fields extracted from either REX, REX2, XOP, VEX, or EVEX fields. 
    // They're globally placed here, in order to avoid testing for each kind of prefix each time. 
    // Instead, one can use the different fields directly from here, regardless the actual encoding mode.
    struct
    {
        ND_UINT32       w:1;                        // REX/REX2/XOP/VEX/EVEX.W
        ND_UINT32       r:1;                        // REX/REX2/XOP/VEX/EVEX.R3 (reg extension)
        ND_UINT32       x:1;                        // REX/REX2/XOP/VEX/EVEX.X3 (index extension)
        ND_UINT32       b:1;                        // REX/REX2/XOP/VEX/EVEX.B3 (base extension)
        ND_UINT32       rp:1;                       // REX2/EVEX.R4 (reg extension, previously known as R')
        ND_UINT32       x4:1;                       // REX2/EVEX.X4 (index extension)
        ND_UINT32       b4:1;                       // REX2/EVEX.B4 (base extension)
        ND_UINT32       p:2;                        // XOP/VEX/EVEX.pp (embedded prefix)
        ND_UINT32       m:5;                        // XOP/VEX/EVEX.mmmmm (decoding table)
        ND_UINT32       l:2;                        // XOP/VEX.L or EVEX.L'L (vector length)
        ND_UINT32       v:4;                        // XOP/VEX/EVEX.VVVV (additional operand)
        ND_UINT32       vp:1;                       // EVEX.V4 (vvvv extension, previously known as V')
        ND_UINT32       bm:1;                       // EVEX.b (embedded broadcast)
        ND_UINT32       z:1;                        // EVEX.z (zero)
        ND_UINT32       k:3;                        // EVEX.aaa (mask registers)
        ND_UINT32       nd:1;                       // EVEX.ND (new data destination)
        ND_UINT32       nf:1;                       // EVEX.NF (no-flags)
        ND_UINT32       sc:4;                       // EVEX.SC0,SC1,SC2,SC3 (standard condition).
    } Exs;

    // Raw instruction components.
    ND_UINT8            Rep;                        // The last rep/repz/repnz prefix. 0 if none.
    ND_UINT8            Seg;                        // The last segment override prefix. 0 if none. FS/GS if 64 bit.
    ND_MODRM            ModRm;                      // ModRM byte.
    ND_SIB              Sib;                        // SIB byte.

    union
    {
        ND_REX          Rex;                        // REX prefix.
        ND_REX2         Rex2;                       // REX2 prefix.
        ND_VEX2         Vex2;                       // VEX 2 prefix.
        ND_VEX3         Vex3;                       // VEX 3 prefix.
        ND_XOP          Xop;                        // XOP prefix.
        ND_EVEX         Evex;                       // EVEX prefix.
    };

    // An Address, Moffset, Displacement or RelativeOffset cannot be present at the same time.
    union
    {
        ND_UINT32       Displacement;               // Displacement. Max 4 bytes. Used in ModRM instructions.
        ND_UINT32       RelativeOffset;             // Relative offset, used for branches. Max 4 bytes.
        ND_UINT64       Moffset;                    // Offset. Used by 'O' operands. It's an absolute address.
        ND_UINT64       AddressNear;                // target64 near address.
        struct
        {
            ND_UINT32   Ip;
            ND_UINT16   Cs;
        } Address;                                  // seg:offset far address.
    };
    ND_UINT64           Immediate1;                 // Can be 8 bytes on x64.
    union
    {
        ND_UINT8        Immediate2;                 // For enter, mainly. Can only be 1 byte.
        ND_UINT8        SseImmediate;               // This immediate actually selects a register.
    };

    ND_UINT8            OperandsCount:4;            // Number of operands (total).
    ND_UINT8            ExpOperandsCount:4;         // Number of explicit operands. Use this if you want to ignore
                                                    // implicit operands such as stack, flags, etc.
    ND_OPERAND          Operands[ND_MAX_OPERAND];   // Instruction operands.

    // SIMD/EVEX information.
    ND_SIMD_EXCEPTIONS  SimdExceptions;             // SIMD Floating-Point Exceptions. Valid only for SIMD instructions!
    ND_UINT8            ExceptionType;              // Exception type. One of ND_EX_TYPE.
    ND_UINT8            TupleType;                  // EVEX tuple type, if EVEX. One of ND_TUPLE.

    // As extracted from the operands themselves.
    ND_UINT8            CsAccess;                   // CS access mode (read/write). Includes only implicit CS accesses.
    ND_UINT8            RipAccess;                  // RIP access mode (read/write).
    ND_UINT8            RflAccess;                  // RFLAGS access mode (read/write), as per the entire register.
    ND_UINT8            StackAccess;                // Stack access mode (push/pop).
    ND_UINT8            MemoryAccess;               // Memory access mode (read/write, including stack or shadow stack).
    ND_FPU_FLAGS        FpuFlagsAccess;             // FPU status word C0-C3 bits access. Valid only for FPU instructions!
    ND_BRANCH_INFO      BranchInfo;                 // Branch information.

    struct
    {
        ND_RFLAGS       Tested;                     // Tested flags.
        ND_RFLAGS       Modified;                   // Modified (according to the result) flags.
        ND_RFLAGS       Set;                        // Flags that are always set to 1.
        ND_RFLAGS       Cleared;                    // Flags that are always cleared to 0.
        ND_RFLAGS       Undefined;                  // Undefined flags.
    } FlagsAccess;

    // Stored inside the instruction entry as well. These are specific for an instruction and do not depend on
    // encoding. Use the flags definitions (ND_FLAG_*, ND_PREF_*, ND_DECO_*, ND_EXOP_*) to access specific bits.
    ND_UINT64           Attributes;                 // Instruction attributes/flags. A collection of ND_FLAG_*.

    // Instruction metadata.
    ND_INS_CLASS        Instruction;                // One of the ND_INS_*
    ND_INS_CATEGORY     Category;                   // One of the ND_CAT_*
    ND_INS_SET          IsaSet;                     // One of the ND_SET_*

    ND_CPUID_FLAG       CpuidFlag;                  // CPUID support flag.
    ND_VALID_MODES      ValidModes;                 // Valid CPU modes for the instruction.
    ND_VALID_PREFIXES   ValidPrefixes;              // Indicates which prefixes are valid for this instruction.
    ND_VALID_DECORATORS ValidDecorators;            // What decorators are accepted by the instruction.

    // Instruction bytes & mnemonic.
    union
    {
        ND_UINT8        PrimaryOpCode;              // Main opcode.
        ND_UINT8        Condition:4;                // Condition code. Valid only if ND_FLAG_COND is set in Attributes.
                                                    // Aliased over low 4 bits inside the main opcode.
    };

#ifndef BDDISASM_NO_MNEMONIC
    const char          *Mnemonic;                  // Instruction mnemonic.
#endif // !BDDISASM_NO_MNEMONIC
    ND_UINT8            InstructionBytes[16];       // The entire instruction.
    ND_UINT8            OpCodeBytes[3];             // Opcode bytes - escape codes and main opcode.

} INSTRUX, *PINSTRUX;



//
// Decoder context. Such a structure must be passed to the NdDecodeWithContext API. This structure must be initialized
// only once, and then it can be re-used across NdDecodeWithContext calls.
//
typedef struct _ND_CONTEXT
{
    ND_UINT64   DefCode : 4;        // Decode mode - one of the ND_CODE_* values.
    ND_UINT64   DefData : 4;        // Data mode - one of the ND_DATA_* values.
    ND_UINT64   DefStack : 4;       // Stack mode - one of the ND_STACK_* values.
    ND_UINT64   VendMode : 4;       // Prefered vendor - one of the ND_VEND_* values.
    ND_UINT64   FeatMode : 8;       // Supported features mask. A combination of ND_FEAT_* values.
    ND_UINT64   Reserved : 40;      // Reserved for future use.
    ND_UINT32   Options;            // Decoding options. A combination of ND_OPTION_* values.
} ND_CONTEXT;

/// Decode only explicit instruction operands. If this options is set, implicit operands, such as RIP or RFLAGS
/// will not be decoded. As a consequence, the following fields inside INSTRUX will be undefined:
/// CsAccess, RipAccess, RflAccess, StackAcces, MemoryAccess, BranchInfo.
#define ND_OPTION_ONLY_EXPLICIT_OPERANDS        0x00000001

// Do NOT zero the output INSTRUX structure before decoding this instruction. Use this option only if the 
// output INSTRUX structure is already zeroed (for example, as is the case when allocating it with calloc).
// Make sure to NOT use this option when making succesive decode calls on the same buffer - this option
// should only be used when decoding an instruction in an output buffer that has just been allocated and
// has been 0-initialized, or if the caller explictly zeroed it before each decode call.
#define ND_OPTION_SKIP_ZERO_INSTRUX             0x00000002


//
// Operands access map. Contains every register except for MSR & XCR, includes memory, flags, RIP, stack.
// Use NdGetFullAccessMap to populate this structure.
//
typedef struct _ND_ACCESS_MAP
{
    ND_UINT8    RipAccess;
    ND_UINT8    FlagsAccess;
    ND_UINT8    StackAccess;
    ND_UINT8    MemAccess;
    ND_UINT8    MxcsrAccess;
    ND_UINT8    PkruAccess;
    ND_UINT8    SspAccess;
    ND_UINT8    GprAccess[ND_MAX_GPR_REGS];
    ND_UINT8    SegAccess[ND_MAX_SEG_REGS];
    ND_UINT8    FpuAccess[ND_MAX_FPU_REGS];
    ND_UINT8    MmxAccess[ND_MAX_MMX_REGS];
    ND_UINT8    SseAccess[ND_MAX_SSE_REGS];
    ND_UINT8    CrAccess [ND_MAX_CR_REGS ];
    ND_UINT8    DrAccess [ND_MAX_DR_REGS ];
    ND_UINT8    TrAccess [ND_MAX_TR_REGS ];
    ND_UINT8    BndAccess[ND_MAX_BND_REGS];
    ND_UINT8    MskAccess[ND_MAX_MSK_REGS];
    ND_UINT8    TmmAccess[ND_MAX_TILE_REGS];
    ND_UINT8    SysAccess[ND_MAX_SYS_REGS];
    ND_UINT8    X87Access[ND_MAX_X87_REGS];
} ND_ACCESS_MAP, *PND_ACCESS_MAP;


//
// Operand reverse-lookup table. Each entry inside this structure contains the pointer to the relevant operand.
// Some rules govern this special structure:
//  - It is not generated by default. The user must call NdGetOperandRlut manually to fill in this structure.
//  - This structure holds pointers inside the INSTRUX provided to the NdGetOperandRlut function; please make sure
//    you call NdGetOperandRlut again if the INSTRUX is relocated, as all the pointers will dangle.
//  - Not all the operand types have a corresponding entry in ND_OPERAND_RLUT, only the usual ones.
//  - Some operands may have multiple entries in ND_OPERAND_RLUT - for example, RMW (read-modify-write) instructions
//    will have Dst1 and Src1 pointing to the same operand.
//  - The implicit registers entries in ND_OPERAND_RLUT will point to the operand which is of that type, and implicit;
//    for example, ND_OPERAND_RLUT.Rax will be NULL for `add rax, rcx`, since in this case, `rax` is not an implicit
//    operand. For `cpuid`, however, ND_OPERAND_RLUT.Rax will point to the implicit `eax` register.
// Use NdGetOperandRlut to populate this structure.
//
typedef struct _ND_OPERAND_RLUT
{
    PND_OPERAND     Dst1;   // First destination operand.
    PND_OPERAND     Dst2;   // Second destination operand.
    PND_OPERAND     Src1;   // First source operand.
    PND_OPERAND     Src2;   // Second source operand.
    PND_OPERAND     Src3;   // Third source operand.
    PND_OPERAND     Src4;   // Fourth source operand.
    PND_OPERAND     Mem1;   // First memory operand.
    PND_OPERAND     Mem2;   // Second memory operand.
    PND_OPERAND     Stack;  // Stack operand.
    PND_OPERAND     Flags;  // Flags register operand.
    PND_OPERAND     Rip;    // Instruction Pointer register operand.
    PND_OPERAND     Cs;     // Implicit CS operand.
    PND_OPERAND     Ss;     // Implicit SS operand.
    PND_OPERAND     Rax;    // Implicit accumulator register operand.
    PND_OPERAND     Rcx;    // Implicit counter register operand.
    PND_OPERAND     Rdx;    // Implicit data register operand
    PND_OPERAND     Rbx;    // Implicit base address register operand.
    PND_OPERAND     Rsp;    // Implicit stack pointer operand.
    PND_OPERAND     Rbp;    // Implicit base pointer operand.
    PND_OPERAND     Rsi;    // Implicit source index operand.
    PND_OPERAND     Rdi;    // Implicit destination index operand.
} ND_OPERAND_RLUT;


#ifdef __cplusplus 
extern "C" {
#endif

//
// Returns the bddisasm version.
//
void
NdGetVersion(
    ND_UINT32 *Major,
    ND_UINT32 *Minor,
    ND_UINT32 *Revision,
    const char **BuildDate,
    const char **BuildTime
    );

//
// Decode one instruction. Note that this is equivalent to: 
// NdDecodeEx(Instrux, Code, ND_MAX_INSTRUCTION_LEN, DefCode, DefData).
// This version should be used if the caller doesn't care about the length of the buffer. Otherwise, use the other
// decode API.
// 
NDSTATUS
NdDecode(
    INSTRUX *Instrux,       // Output decoded instruction.
    const ND_UINT8 *Code,    // Buffer containing the instruction bytes.
    ND_UINT8 DefCode,        // Decode mode - one of the ND_CODE_* values.
    ND_UINT8 DefData         // Data mode - one of the ND_DATA_* value.
    );

//
// Decode one instruction. Note that this is equivalent to: 
// NdDecodeEx2(Instrux, Code, Size, DefCode, DefData, DefCode, ND_VEND_ANY).
// By default, the used vendor will be ND_VEND_ANY, so all instructions will be decoded.
// By default, the feature mode will be ND_FEAT_ALL, so all instructions will be decoded (but may yield error where
// otherwise a NOP would be encoded - use ND_FEAT_NONE in that case).
// 
NDSTATUS
NdDecodeEx(
    INSTRUX *Instrux,       // Output decoded instruction.
    const ND_UINT8 *Code,      // Buffer containing the instruction bytes.
    ND_SIZET Size,          // Maximum size of the Code buffer.
    ND_UINT8 DefCode,          // Decode mode - one of the ND_CODE_* values.
    ND_UINT8 DefData           // Data mode - one of the ND_DATA_* value.
    );

//
// Fills a ND_CONTEXT structure, and calls NdDecodeWithContext. The feature mode will be ND_FEAT_ALL by default.
//
NDSTATUS
NdDecodeEx2(
    INSTRUX *Instrux,       // Output decoded instruction.
    const ND_UINT8 *Code,      // Buffer containing the instruction bytes.
    ND_SIZET Size,          // Maximum size of the Code buffer.
    ND_UINT8 DefCode,          // Decode mode - one of the ND_CODE_* values.
    ND_UINT8 DefData,          // Data mode - one of the ND_DATA_* value.
    ND_UINT8 DefStack,         // Stack mode - one of the ND_STACK_* values.
    ND_UINT8 PreferedVendor    // Preferred vendor - one of the ND_VEND_* values.
    );

//
// This API received a decode context, where it expects DefCode, DefData, DefStack, VendMode and FeatMode to be 
// already initialized. The Context will not be modified by the decoder, so it can be reused across decode calls.
// The Context should initially be initialized using NdInitContext. This will ensure backwards compatibility
// by setting new fields to default values.
// Note that this is the base decoding API, and this ends up being called by all the other decoding APIs, after 
// providing default arguments and filling them in the Context structure. For maximum speed, use this instead of
// the others.
//
NDSTATUS
NdDecodeWithContext(
    INSTRUX *Instrux,       // Output decoded instruction.
    const ND_UINT8 *Code,      // Buffer containing the instruction bytes.
    ND_SIZET Size,          // Maximum size of the Code buffer.
    ND_CONTEXT *Context     // Context describing decode mode, vendor mode and supported features.
    );

//
// Convert the given instruction into textual representation (Intel syntax).
//
NDSTATUS
NdToText(
    const INSTRUX *Instrux,
    ND_UINT64 Rip,
    ND_UINT32 BufferSize,
    char *Buffer
    );

//
// Returns true if the instruction is RIP relative. Note that this function is kept for backwards compatibility, since
// there already is a IsRipRelative field inside INSTRUX.
//
ND_BOOL
NdIsInstruxRipRelative(
    const INSTRUX *Instrux
    );

//
// Returns an access map that contains the access for each register.
//
NDSTATUS
NdGetFullAccessMap(
    const INSTRUX *Instrux,
    ND_ACCESS_MAP *AccessMap
    );

//
// Returns an operand reverse-lookup. One can use the Rlut to quickly reference different kinds of operands in INSTRUX.
//
NDSTATUS
NdGetOperandRlut(
    const INSTRUX *Instrux,
    ND_OPERAND_RLUT *Rlut
    );

//
// Initialize the decoder context.
//
void
NdInitContext(
    ND_CONTEXT *Context
    );

#ifdef __cplusplus 
}
#endif

// #pragma warning(default: 4214) // Bitfield in type other than int.
// #pragma warning(default: 4201) // Nonstandard extension used: nameless struct/union.
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif // BDX86_CORE_H
