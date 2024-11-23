/*
 * Copyright (c) 2020 Bitdefender
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef BDDISASM_STATUS_H
#define BDDISASM_STATUS_H

//
// Return statuses.
//
typedef ND_UINT32 NDSTATUS;

// Success codes are all < 0x80000000.
#define ND_STATUS_SUCCESS                               0x00000000 // All good.

// Hint/success codes.
#define ND_STATUS_HINT_OPERAND_NOT_USED                 0x00000001

// Error codes are all > 0x80000000.
#define ND_STATUS_BUFFER_TOO_SMALL                      0x80000001 // The provided input buffer is too small.
#define ND_STATUS_INVALID_ENCODING                      0x80000002 // Invalid encoding/instruction.
#define ND_STATUS_INSTRUCTION_TOO_LONG                  0x80000003 // Instruction exceeds the maximum 15 bytes.
#define ND_STATUS_INVALID_PREFIX_SEQUENCE               0x80000004 // Invalid prefix sequence is present.
#define ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION       0x80000005 // The instruction uses an invalid register.
#define ND_STATUS_XOP_WITH_PREFIX                       0x80000006 // XOP is present, but also a legacy prefix.
#define ND_STATUS_VEX_WITH_PREFIX                       0x80000007 // VEX is present, but also a legacy prefix.
#define ND_STATUS_EVEX_WITH_PREFIX                      0x80000008 // EVEX is present, but also a legacy prefix.
#define ND_STATUS_INVALID_ENCODING_IN_MODE              0x80000009 // Invalid encoding/instruction.
#define ND_STATUS_BAD_LOCK_PREFIX                       0x8000000A // Invalid usage of LOCK.
#define ND_STATUS_CS_LOAD                               0x8000000B // An attempt to load the CS register.
#define ND_STATUS_66_NOT_ACCEPTED                       0x8000000C // 0x66 prefix is not accepted.
#define ND_STATUS_16_BIT_ADDRESSING_NOT_SUPPORTED       0x8000000D // 16 bit addressing mode not supported.
#define ND_STATUS_RIP_REL_ADDRESSING_NOT_SUPPORTED      0x8000000E // RIP-relative addressing not supported.

// VEX/EVEX specific errors.
#define ND_STATUS_VSIB_WITHOUT_SIB                      0x80000030 // Instruction uses VSIB, but SIB is not present.
#define ND_STATUS_INVALID_VSIB_REGS                     0x80000031 // VSIB addressing, same vector reg used more than once.
#define ND_STATUS_VEX_VVVV_MUST_BE_ZERO                 0x80000032 // VEX.VVVV field must be zero.
#define ND_STATUS_MASK_NOT_SUPPORTED                    0x80000033 // Masking is not supported.
#define ND_STATUS_MASK_REQUIRED                         0x80000034 // Masking is mandatory.
#define ND_STATUS_ER_SAE_NOT_SUPPORTED                  0x80000035 // Embedded rounding/SAE not supported.
#define ND_STATUS_ZEROING_NOT_SUPPORTED                 0x80000036 // Zeroing not supported.
#define ND_STATUS_ZEROING_ON_MEMORY                     0x80000037 // Zeroing on memory.
#define ND_STATUS_ZEROING_NO_MASK                       0x80000038 // Zeroing without masking.
#define ND_STATUS_BROADCAST_NOT_SUPPORTED               0x80000039 // Broadcast not supported.
#define ND_STATUS_BAD_EVEX_V_PRIME                      0x80000040 // EVEX.V' field must be one (negated 0).
#define ND_STATUS_BAD_EVEX_LL                           0x80000041 // EVEX.L'L field is invalid for the instruction.
#define ND_STATUS_SIBMEM_WITHOUT_SIB                    0x80000042 // Instruction uses SIBMEM, but SIB is not present.
#define ND_STATUS_INVALID_TILE_REGS                     0x80000043 // Tile registers are not unique.
#define ND_STATUS_INVALID_DEST_REGS                     0x80000044 // Destination register is not unique (used as src).
#define ND_STATUS_INVALID_EVEX_BYTE3                    0x80000045 // EVEX payload byte 3 is invalid.
#define ND_STATUS_BAD_EVEX_U                            0x80000046 // EVEX.U field is invalid.


// Not encoding specific.
#define ND_STATUS_INVALID_PARAMETER                     0x80000100 // An invalid parameter was provided.
#define ND_STATUS_INVALID_INSTRUX                       0x80000101 // The INSTRUX contains unexpected values.
#define ND_STATUS_BUFFER_OVERFLOW                       0x80000103 // Not enough space is available to format textual disasm.

#define ND_STATUS_INTERNAL_ERROR                        0x80000200 // Internal error occurred.


#define ND_SUCCESS(status)                              (status < 0x80000000)

#endif // BDDISASM_STATUS_H
