//===- opcode-map.cpp - Transpiler ----------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "opcode-map.h"

#include "amdgpu-mc-tables.h"

#include <cassert>
#include <optional>

// AMDGPU target-private headers.
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"

using namespace llvm;

namespace COMGR::transpiler {

namespace {

// Opcode named by an AMDGPU InstrMapping helper result, or nullopt for either
// way the helpers signal "no mapping": -1 and INSTRUCTION_LIST_END.
std::optional<unsigned> mappedOpcode(int Result) {
  if (Result <= 0 || Result >= AMDGPU::INSTRUCTION_LIST_END)
    return std::nullopt;
  return Result;
}

// One kCanonTable row: a canonical AMDGPU pseudo opcode and its CanonicalOp.
struct Entry {
  unsigned Opc;
  CanonicalOp Sem;
};

#define E(OP, SEM)                                                             \
  Entry { AMDGPU::OP, CanonicalOp::SEM }

#define BUFFER_RAW(OP, SEM) E(OP##_OFFSET, SEM), E(OP##_OFFEN, SEM)

// DS pseudos come in a pre-GFX9 form carrying an M0 operand and a GFX9-and-
// later form without one. Both reach the same CanonicalOp.
#define DS_M0(OP, SEM) E(OP, SEM), E(OP##_gfx9, SEM)

static const Entry kCanonTable[] = {
    // clang-format off
    E(S_MOV_B32, S_MOV_B32),
    E(S_MOV_B64, S_MOV_B64),
    E(S_MOVRELS_B32, S_MOVRELS_B32),
    E(S_MOVRELS_B64, S_MOVRELS_B64),
    E(S_MOVRELD_B32, S_MOVRELD_B32),
    E(S_MOVRELD_B64, S_MOVRELD_B64),
    E(S_MOVRELSD_2_B32, S_MOVRELSD_2_B32),
    E(S_CMOV_B32, S_CMOV_B32),
    E(S_CMOV_B64, S_CMOV_B64),
    E(S_BREV_B32, S_BREV_B32),
    E(S_BREV_B64, S_BREV_B64),
    E(S_NOT_B32, S_NOT_B32),
    E(S_NOT_B64, S_NOT_B64),
    E(S_CEIL_F32, S_CEIL_F32),
    E(S_FLOOR_F32, S_FLOOR_F32),
    E(S_TRUNC_F32, S_TRUNC_F32),
    E(S_RNDNE_F32, S_RNDNE_F32),
    E(S_CVT_F32_I32, S_CVT_F32_I32),
    E(S_CVT_F32_U32, S_CVT_F32_U32),
    E(S_CVT_I32_F32, S_CVT_I32_F32),
    E(S_CVT_U32_F32, S_CVT_U32_F32),
    E(S_CVT_F16_F32, S_CVT_F16_F32),
    E(S_CVT_F32_F16, S_CVT_F32_F16),
    E(S_CVT_HI_F32_F16, S_CVT_HI_F32_F16),
    E(S_CEIL_F16, S_CEIL_F16),
    E(S_FLOOR_F16, S_FLOOR_F16),
    E(S_TRUNC_F16, S_TRUNC_F16),
    E(S_RNDNE_F16, S_RNDNE_F16),
    E(S_FF1_I32_B32, S_FF1_I32_B32),
    E(S_FF1_I32_B64, S_FF1_I32_B64),
    E(S_FLBIT_I32_B32, S_FLBIT_I32_B32),
    E(S_FLBIT_I32_B64, S_FLBIT_I32_B64),
    E(S_FLBIT_I32, S_FLBIT_I32),
    E(S_FLBIT_I32_I64, S_FLBIT_I32_I64),
    E(S_SEXT_I32_I8, S_SEXT_I32_I8),
    E(S_SEXT_I32_I16, S_SEXT_I32_I16),
    E(S_BITSET0_B32, S_BITSET0_B32),
    E(S_BITSET0_B64, S_BITSET0_B64),
    E(S_BITSET1_B32, S_BITSET1_B32),
    E(S_BITSET1_B64, S_BITSET1_B64),
    E(S_BITREPLICATE_B64_B32, S_BITREPLICATE_B64_B32),
    E(S_ABS_I32, S_ABS_I32),
    E(S_BCNT0_I32_B32, S_BCNT0_I32_B32),
    E(S_BCNT0_I32_B64, S_BCNT0_I32_B64),
    E(S_BCNT1_I32_B32, S_BCNT1_I32_B32),
    E(S_BCNT1_I32_B64, S_BCNT1_I32_B64),
    E(S_QUADMASK_B32, S_QUADMASK_B32),
    E(S_QUADMASK_B64, S_QUADMASK_B64),
    E(S_WQM_B32, S_WQM_B32),
    E(S_WQM_B64, S_WQM_B64),
    E(S_AND_SAVEEXEC_B32, S_AND_SAVEEXEC_B32),
    E(S_AND_SAVEEXEC_B64, S_AND_SAVEEXEC_B64),
    E(S_OR_SAVEEXEC_B32, S_OR_SAVEEXEC_B32),
    E(S_OR_SAVEEXEC_B64, S_OR_SAVEEXEC_B64),
    E(S_XOR_SAVEEXEC_B32, S_XOR_SAVEEXEC_B32),
    E(S_XOR_SAVEEXEC_B64, S_XOR_SAVEEXEC_B64),
    E(S_NAND_SAVEEXEC_B32, S_NAND_SAVEEXEC_B32),
    E(S_NAND_SAVEEXEC_B64, S_NAND_SAVEEXEC_B64),
    E(S_NOR_SAVEEXEC_B32, S_NOR_SAVEEXEC_B32),
    E(S_NOR_SAVEEXEC_B64, S_NOR_SAVEEXEC_B64),
    E(S_XNOR_SAVEEXEC_B32, S_XNOR_SAVEEXEC_B32),
    E(S_XNOR_SAVEEXEC_B64, S_XNOR_SAVEEXEC_B64),
    E(S_ANDN1_SAVEEXEC_B32, S_ANDN1_SAVEEXEC_B32),
    E(S_ANDN1_SAVEEXEC_B64, S_ANDN1_SAVEEXEC_B64),
    E(S_ORN1_SAVEEXEC_B32, S_ORN1_SAVEEXEC_B32),
    E(S_ORN1_SAVEEXEC_B64, S_ORN1_SAVEEXEC_B64),
    E(S_ANDN2_SAVEEXEC_B32, S_ANDN2_SAVEEXEC_B32),
    E(S_ANDN2_SAVEEXEC_B64, S_ANDN2_SAVEEXEC_B64),
    E(S_ORN2_SAVEEXEC_B32, S_ORN2_SAVEEXEC_B32),
    E(S_ORN2_SAVEEXEC_B64, S_ORN2_SAVEEXEC_B64),
    E(S_ANDN1_WREXEC_B32, S_ANDN1_WREXEC_B32),
    E(S_ANDN1_WREXEC_B64, S_ANDN1_WREXEC_B64),
    E(S_ANDN2_WREXEC_B32, S_ANDN2_WREXEC_B32),
    E(S_ANDN2_WREXEC_B64, S_ANDN2_WREXEC_B64),
    E(S_GETPC_B64, S_GETPC_B64),
    E(S_SETPC_B64, S_SETPC_B64),
    E(S_SWAPPC_B64, S_SWAPPC_B64),
    E(S_ADD_PC_I64, S_ADD_PC_I64),
    E(S_RFE_B64, S_RFE_B64),
    E(S_BARRIER_SIGNAL_IMM, S_BARRIER_SIGNAL_IMM),
    E(S_BARRIER_SIGNAL_M0, S_BARRIER_SIGNAL_M0),
    E(S_BARRIER_SIGNAL_ISFIRST_IMM, S_BARRIER_SIGNAL_ISFIRST_IMM),
    E(S_BARRIER_SIGNAL_ISFIRST_M0, S_BARRIER_SIGNAL_ISFIRST_M0),
    E(S_GET_BARRIER_STATE_IMM, S_GET_BARRIER_STATE_IMM),
    E(S_GET_BARRIER_STATE_M0, S_GET_BARRIER_STATE_M0),
    E(S_BARRIER_INIT_IMM, S_BARRIER_INIT_IMM),
    E(S_BARRIER_INIT_M0, S_BARRIER_INIT_M0),
    E(S_BARRIER_JOIN_IMM, S_BARRIER_JOIN_IMM),
    E(S_BARRIER_JOIN_M0, S_BARRIER_JOIN_M0),
    E(S_WAKEUP_BARRIER_IMM, S_WAKEUP_BARRIER_IMM),
    E(S_WAKEUP_BARRIER_M0, S_WAKEUP_BARRIER_M0),
    E(S_GET_SHADER_CYCLES_U64, S_GET_SHADER_CYCLES_U64),
    E(S_SENDMSG_RTN_B32, S_SENDMSG_RTN_B32),
    E(S_SENDMSG_RTN_B64, S_SENDMSG_RTN_B64),
    E(S_ALLOC_VGPR, S_ALLOC_VGPR),
    E(S_SLEEP_VAR, S_SLEEP_VAR),
    E(S_ADD_U32, S_ADD_U32),
    E(S_ADD_I32, S_ADD_I32),
    E(S_ADDC_U32, S_ADDC_U32),
    E(S_SUB_U32, S_SUB_U32),
    E(S_SUB_I32, S_SUB_I32),
    E(S_SUBB_U32, S_SUBB_U32),
    E(S_AND_B32, S_AND_B32),
    E(S_AND_B64, S_AND_B64),
    E(S_OR_B32, S_OR_B32),
    E(S_OR_B64, S_OR_B64),
    E(S_XOR_B32, S_XOR_B32),
    E(S_XOR_B64, S_XOR_B64),
    E(S_ANDN2_B32, S_ANDN2_B32),
    E(S_ANDN2_B64, S_ANDN2_B64),
    E(S_ORN2_B32, S_ORN2_B32),
    E(S_ORN2_B64, S_ORN2_B64),
    E(S_NAND_B32, S_NAND_B32),
    E(S_NAND_B64, S_NAND_B64),
    E(S_NOR_B32, S_NOR_B32),
    E(S_NOR_B64, S_NOR_B64),
    E(S_XNOR_B32, S_XNOR_B32),
    E(S_XNOR_B64, S_XNOR_B64),
    E(S_ABSDIFF_I32, S_ABSDIFF_I32),
    E(S_LSHL_B32, S_LSHL_B32),
    E(S_LSHL_B64, S_LSHL_B64),
    E(S_LSHR_B32, S_LSHR_B32),
    E(S_LSHR_B64, S_LSHR_B64),
    E(S_ASHR_I32, S_ASHR_I32),
    E(S_ASHR_I64, S_ASHR_I64),
    E(S_MUL_I32, S_MUL_I32),
    E(S_MUL_HI_U32, S_MUL_HI_U32),
    E(S_MUL_HI_I32, S_MUL_HI_I32),
    E(S_MUL_U64, S_MUL_U64),
    E(S_MUL_F32, S_MUL_F32),
    E(S_FMAC_F32, S_FMAC_F32),
    E(S_FMAAK_F32, S_FMAAK_F32),
    E(S_BFE_U32, S_BFE_U32),
    E(S_BFE_I32, S_BFE_I32),
    E(S_BFE_I64, S_BFE_I64),
    E(S_BFM_B32, S_BFM_B32),
    E(S_BFM_B64, S_BFM_B64),
    E(S_CSELECT_B32, S_CSELECT_B32),
    E(S_CSELECT_B64, S_CSELECT_B64),
    E(S_MIN_I32, S_MIN_I32),
    E(S_MIN_U32, S_MIN_U32),
    E(S_MAX_I32, S_MAX_I32),
    E(S_MAX_U32, S_MAX_U32),
    E(S_PACK_LL_B32_B16, S_PACK_LL_B32_B16),
    E(S_PACK_LH_B32_B16, S_PACK_LH_B32_B16),
    E(S_LSHL1_ADD_U32, S_LSHL1_ADD_U32),
    E(S_LSHL2_ADD_U32, S_LSHL2_ADD_U32),
    E(S_LSHL3_ADD_U32, S_LSHL3_ADD_U32),
    E(S_LSHL4_ADD_U32, S_LSHL4_ADD_U32),
    // gfx12 renamed the assembly mnemonics but retained these pseudos.
    E(S_ADD_U64, S_ADD_NC_U64),
    E(S_SUB_U64, S_SUB_NC_U64),
    E(S_CMP_EQ_U32, S_CMP_EQ_U32),
    E(S_CMP_LG_U32, S_CMP_LG_U32),
    E(S_CMP_GT_U32, S_CMP_GT_U32),
    E(S_CMP_GE_U32, S_CMP_GE_U32),
    E(S_CMP_LT_U32, S_CMP_LT_U32),
    E(S_CMP_LE_U32, S_CMP_LE_U32),
    E(S_CMP_EQ_I32, S_CMP_EQ_I32),
    E(S_CMP_LG_I32, S_CMP_LG_I32),
    E(S_CMP_GT_I32, S_CMP_GT_I32),
    E(S_CMP_GE_I32, S_CMP_GE_I32),
    E(S_CMP_LT_I32, S_CMP_LT_I32),
    E(S_CMP_LE_I32, S_CMP_LE_I32),
    E(S_CMP_EQ_U64, S_CMP_EQ_U64),
    E(S_CMP_LG_U64, S_CMP_LG_U64),
    E(S_CMP_EQ_F32, S_CMP_EQ_F32),
    E(S_CMP_LG_F32, S_CMP_LG_F32),
    E(S_CMP_GT_F32, S_CMP_GT_F32),
    E(S_CMP_GE_F32, S_CMP_GE_F32),
    E(S_CMP_LT_F32, S_CMP_LT_F32),
    E(S_CMP_LE_F32, S_CMP_LE_F32),
    E(S_CMP_NEQ_F32, S_CMP_NEQ_F32),
    E(S_CMP_NGT_F32, S_CMP_NGT_F32),
    E(S_CMP_NGE_F32, S_CMP_NGE_F32),
    E(S_CMP_NLT_F32, S_CMP_NLT_F32),
    E(S_CMP_NLE_F32, S_CMP_NLE_F32),
    E(S_CMP_NLG_F32, S_CMP_NLG_F32),
    E(S_CMP_O_F32, S_CMP_O_F32),
    E(S_CMP_U_F32, S_CMP_U_F32),
    E(S_CMP_EQ_F16, S_CMP_EQ_F16),
    E(S_CMP_LG_F16, S_CMP_LG_F16),
    E(S_CMP_GT_F16, S_CMP_GT_F16),
    E(S_CMP_GE_F16, S_CMP_GE_F16),
    E(S_CMP_LT_F16, S_CMP_LT_F16),
    E(S_CMP_LE_F16, S_CMP_LE_F16),
    E(S_CMP_NEQ_F16, S_CMP_NEQ_F16),
    E(S_CMP_NGT_F16, S_CMP_NGT_F16),
    E(S_CMP_NGE_F16, S_CMP_NGE_F16),
    E(S_CMP_NLT_F16, S_CMP_NLT_F16),
    E(S_CMP_NLE_F16, S_CMP_NLE_F16),
    E(S_CMP_NLG_F16, S_CMP_NLG_F16),
    E(S_CMP_O_F16, S_CMP_O_F16),
    E(S_CMP_U_F16, S_CMP_U_F16),
    E(S_BITCMP0_B32, S_BITCMP0_B32),
    E(S_BITCMP1_B32, S_BITCMP1_B32),
    E(S_BITCMP0_B64, S_BITCMP0_B64),
    E(S_BITCMP1_B64, S_BITCMP1_B64),
    E(S_MOVK_I32, S_MOVK_I32),
    E(S_ADDK_I32, S_ADDK_I32),
    E(S_MULK_I32, S_MULK_I32),
    E(S_CMPK_EQ_I32, S_CMPK_EQ_I32),
    E(S_CMPK_LG_I32, S_CMPK_LG_I32),
    E(S_CMPK_GT_I32, S_CMPK_GT_I32),
    E(S_CMPK_GE_I32, S_CMPK_GE_I32),
    E(S_CMPK_LT_I32, S_CMPK_LT_I32),
    E(S_CMPK_LE_I32, S_CMPK_LE_I32),
    E(S_CMPK_EQ_U32, S_CMPK_EQ_U32),
    E(S_CMPK_LG_U32, S_CMPK_LG_U32),
    E(S_CMPK_GT_U32, S_CMPK_GT_U32),
    E(S_CMPK_GE_U32, S_CMPK_GE_U32),
    E(S_CMPK_LT_U32, S_CMPK_LT_U32),
    E(S_CMPK_LE_U32, S_CMPK_LE_U32),
    E(S_GETREG_B32, S_GETREG_B32),
    E(S_SETREG_B32, S_SETREG_B32),
    E(S_SETREG_IMM32_B32, S_SETREG_IMM32_B32),
    E(S_ENDPGM, S_ENDPGM),
    E(S_ENDPGM_SAVED, S_ENDPGM_SAVED),
    E(S_WAITCNT, S_WAITCNT),
    E(S_WAIT_LOADCNT, S_WAIT_LOADCNT),
    E(S_WAIT_STORECNT, S_WAIT_STORECNT),
    E(S_WAIT_DSCNT, S_WAIT_DSCNT),
    E(S_WAIT_KMCNT, S_WAIT_KMCNT),
    E(S_WAIT_EXPCNT, S_WAIT_EXPCNT),
    E(S_WAIT_SAMPLECNT, S_WAIT_SAMPLECNT),
    E(S_WAIT_BVHCNT, S_WAIT_BVHCNT),
    E(S_WAIT_EVENT, S_WAIT_EVENT),
    E(S_WAIT_LOADCNT_DSCNT, S_WAIT_LOADCNT_DSCNT),
    E(S_WAIT_STORECNT_DSCNT, S_WAIT_STORECNT_DSCNT),
    E(S_WAIT_IDLE, S_WAIT_IDLE),
    E(S_WAIT_ASYNCCNT, S_WAIT_ASYNCCNT),
    E(S_WAIT_TENSORCNT, S_WAIT_TENSORCNT),
    E(S_WAIT_XCNT, S_WAIT_XCNT),
    // gfx12 renamed the mnemonic to `s_wait_alu`, but the pseudo LLVM keys on
    // still carries the original `S_WAITCNT_DEPCTR` spelling.
    E(S_WAITCNT_DEPCTR, S_WAIT_ALU),
    E(S_NOP, S_NOP),
    E(S_CLAUSE, S_CLAUSE),
    E(S_DELAY_ALU, S_DELAY_ALU),
    E(S_SLEEP, S_SLEEP),
    E(S_SETPRIO, S_SETPRIO),
    E(S_SETHALT, S_SETHALT),
    E(S_MONITOR_SLEEP, S_MONITOR_SLEEP),
    E(S_WAKEUP, S_WAKEUP),
    E(S_SETPRIO_INC_WG, S_SETPRIO_INC_WG),
    E(S_SET_VGPR_MSB, S_SET_VGPR_MSB),
    E(S_CODE_END, S_CODE_END),
    E(S_INCPERFLEVEL, S_INCPERFLEVEL),
    E(S_DECPERFLEVEL, S_DECPERFLEVEL),
    E(S_TTRACEDATA, S_TTRACEDATA),
    E(S_TTRACEDATA_IMM, S_TTRACEDATA_IMM),
    E(S_ICACHE_INV, S_ICACHE_INV),
    E(S_BARRIER_WAIT, S_BARRIER_WAIT),
    E(S_BARRIER_LEAVE, S_BARRIER_LEAVE),
    E(S_BRANCH, S_BRANCH),
    E(S_CBRANCH_SCC0, S_CBRANCH_SCC0),
    E(S_CBRANCH_SCC1, S_CBRANCH_SCC1),
    E(S_CBRANCH_VCCZ, S_CBRANCH_VCCZ),
    E(S_CBRANCH_VCCNZ, S_CBRANCH_VCCNZ),
    E(S_CBRANCH_EXECZ, S_CBRANCH_EXECZ),
    E(S_CBRANCH_EXECNZ, S_CBRANCH_EXECNZ),
    E(S_TRAP, S_TRAP),
    E(S_ROUND_MODE, S_ROUND_MODE),
    E(S_DENORM_MODE, S_DENORM_MODE),
    E(S_SENDMSG, S_SENDMSG),
    E(S_SENDMSGHALT, S_SENDMSGHALT),
    E(S_LOAD_I8_IMM, S_LOAD_I8),
    E(S_LOAD_I8_SGPR, S_LOAD_I8),
    E(S_LOAD_I8_SGPR_IMM, S_LOAD_I8),
    E(S_LOAD_U8_IMM, S_LOAD_U8),
    E(S_LOAD_U8_SGPR, S_LOAD_U8),
    E(S_LOAD_U8_SGPR_IMM, S_LOAD_U8),
    E(S_LOAD_I16_IMM, S_LOAD_I16),
    E(S_LOAD_I16_SGPR, S_LOAD_I16),
    E(S_LOAD_I16_SGPR_IMM, S_LOAD_I16),
    E(S_LOAD_U16_IMM, S_LOAD_U16),
    E(S_LOAD_U16_SGPR, S_LOAD_U16),
    E(S_LOAD_U16_SGPR_IMM, S_LOAD_U16),
    E(S_LOAD_DWORD_IMM, S_LOAD_B32),
    E(S_LOAD_DWORD_SGPR, S_LOAD_B32),
    E(S_LOAD_DWORD_SGPR_IMM, S_LOAD_B32),
    E(S_LOAD_DWORDX2_IMM, S_LOAD_B64),
    E(S_LOAD_DWORDX2_SGPR, S_LOAD_B64),
    E(S_LOAD_DWORDX2_SGPR_IMM, S_LOAD_B64),
    E(S_LOAD_DWORDX3_IMM, S_LOAD_B96),
    E(S_LOAD_DWORDX3_SGPR, S_LOAD_B96),
    E(S_LOAD_DWORDX3_SGPR_IMM, S_LOAD_B96),
    E(S_LOAD_DWORDX4_IMM, S_LOAD_B128),
    E(S_LOAD_DWORDX4_SGPR, S_LOAD_B128),
    E(S_LOAD_DWORDX4_SGPR_IMM, S_LOAD_B128),
    E(S_LOAD_DWORDX8_IMM, S_LOAD_B256),
    E(S_LOAD_DWORDX8_SGPR, S_LOAD_B256),
    E(S_LOAD_DWORDX8_SGPR_IMM, S_LOAD_B256),
    E(S_LOAD_DWORDX16_IMM, S_LOAD_B512),
    E(S_LOAD_DWORDX16_SGPR, S_LOAD_B512),
    E(S_LOAD_DWORDX16_SGPR_IMM, S_LOAD_B512),
    E(GLOBAL_LOAD_DWORD, GLOBAL_LOAD_B32),
    E(GLOBAL_LOAD_DWORDX2, GLOBAL_LOAD_B64),
    E(GLOBAL_LOAD_DWORDX3, GLOBAL_LOAD_B96),
    E(GLOBAL_LOAD_DWORDX4, GLOBAL_LOAD_B128),
    E(GLOBAL_LOAD_SBYTE, GLOBAL_LOAD_I8),
    E(GLOBAL_LOAD_SSHORT, GLOBAL_LOAD_I16),
    E(GLOBAL_LOAD_UBYTE, GLOBAL_LOAD_U8),
    E(GLOBAL_LOAD_USHORT, GLOBAL_LOAD_U16),
    E(GLOBAL_STORE_BYTE, GLOBAL_STORE_B8),
    E(GLOBAL_STORE_BYTE_D16_HI, GLOBAL_STORE_D16_HI_B8),
    E(GLOBAL_STORE_DWORD, GLOBAL_STORE_B32),
    E(GLOBAL_STORE_DWORDX2, GLOBAL_STORE_B64),
    E(GLOBAL_STORE_DWORDX3, GLOBAL_STORE_B96),
    E(GLOBAL_STORE_DWORDX4, GLOBAL_STORE_B128),
    E(GLOBAL_STORE_SHORT, GLOBAL_STORE_B16),
    E(GLOBAL_STORE_SHORT_D16_HI, GLOBAL_STORE_D16_HI_B16),
    E(GLOBAL_ATOMIC_ADD, GLOBAL_ATOMIC_ADD_U32),
    E(GLOBAL_ATOMIC_ADD_RTN, GLOBAL_ATOMIC_ADD_RTN_U32),
    E(GLOBAL_ATOMIC_ADD_RTN_agpr, GLOBAL_ATOMIC_ADD_RTN_U32),
    BUFFER_RAW(BUFFER_LOAD_DWORD, BUFFER_LOAD_B32),
    BUFFER_RAW(BUFFER_LOAD_DWORDX2, BUFFER_LOAD_B64),
    BUFFER_RAW(BUFFER_LOAD_DWORDX3, BUFFER_LOAD_B96),
    BUFFER_RAW(BUFFER_LOAD_DWORDX4, BUFFER_LOAD_B128),
    BUFFER_RAW(BUFFER_LOAD_SBYTE, BUFFER_LOAD_I8),
    BUFFER_RAW(BUFFER_LOAD_SBYTE_D16, BUFFER_LOAD_D16_I8),
    BUFFER_RAW(BUFFER_LOAD_SBYTE_D16_HI, BUFFER_LOAD_D16_HI_I8),
    BUFFER_RAW(BUFFER_LOAD_SHORT_D16, BUFFER_LOAD_D16_B16),
    BUFFER_RAW(BUFFER_LOAD_SHORT_D16_HI, BUFFER_LOAD_D16_HI_B16),
    BUFFER_RAW(BUFFER_LOAD_SSHORT, BUFFER_LOAD_I16),
    BUFFER_RAW(BUFFER_LOAD_UBYTE, BUFFER_LOAD_U8),
    BUFFER_RAW(BUFFER_LOAD_UBYTE_D16, BUFFER_LOAD_D16_U8),
    BUFFER_RAW(BUFFER_LOAD_UBYTE_D16_HI, BUFFER_LOAD_D16_HI_U8),
    BUFFER_RAW(BUFFER_LOAD_USHORT, BUFFER_LOAD_U16),
    BUFFER_RAW(BUFFER_STORE_BYTE, BUFFER_STORE_B8),
    BUFFER_RAW(BUFFER_STORE_BYTE_D16_HI, BUFFER_STORE_D16_HI_B8),
    BUFFER_RAW(BUFFER_STORE_DWORD, BUFFER_STORE_B32),
    BUFFER_RAW(BUFFER_STORE_DWORDX2, BUFFER_STORE_B64),
    BUFFER_RAW(BUFFER_STORE_DWORDX3, BUFFER_STORE_B96),
    BUFFER_RAW(BUFFER_STORE_DWORDX4, BUFFER_STORE_B128),
    BUFFER_RAW(BUFFER_STORE_SHORT, BUFFER_STORE_B16),
    BUFFER_RAW(BUFFER_STORE_SHORT_D16_HI, BUFFER_STORE_D16_HI_B16),
    E(DS_BPERMUTE_B32, DS_BPERMUTE_B32),
    E(DS_BPERMUTE_FI_B32, DS_BPERMUTE_FI_B32),
    E(DS_PERMUTE_B32, DS_PERMUTE_B32),
    E(DS_SWIZZLE_B32, DS_SWIZZLE_B32),
    E(DS_LOAD_TR16_B128, DS_LOAD_TR16_B128),
    E(DS_LOAD_TR8_B64, DS_LOAD_TR8_B64),
    DS_M0(DS_READ_U8, DS_LOAD_U8),
    DS_M0(DS_READ_I8, DS_LOAD_I8),
    DS_M0(DS_READ_U16, DS_LOAD_U16),
    DS_M0(DS_READ_I16, DS_LOAD_I16),
    DS_M0(DS_READ_B32, DS_LOAD_B32),
    DS_M0(DS_READ_B64, DS_LOAD_B64),
    DS_M0(DS_READ_B96, DS_LOAD_B96),
    DS_M0(DS_READ_B128, DS_LOAD_B128),
    DS_M0(DS_WRITE_B8, DS_STORE_B8),
    DS_M0(DS_WRITE_B16, DS_STORE_B16),
    DS_M0(DS_WRITE_B32, DS_STORE_B32),
    DS_M0(DS_WRITE_B64, DS_STORE_B64),
    DS_M0(DS_WRITE_B96, DS_STORE_B96),
    DS_M0(DS_WRITE_B128, DS_STORE_B128),
    // The partial stores have a single pseudo each, with no GFX9 counterpart.
    E(DS_WRITE_B8_D16_HI, DS_STORE_B8_D16_HI),
    E(DS_WRITE_B16_D16_HI, DS_STORE_B16_D16_HI),
    DS_M0(DS_READ2_B32, DS_LOAD_2ADDR_B32),
    DS_M0(DS_READ2_B64, DS_LOAD_2ADDR_B64),
    DS_M0(DS_READ2ST64_B32, DS_LOAD_2ADDR_STRIDE64_B32),
    DS_M0(DS_READ2ST64_B64, DS_LOAD_2ADDR_STRIDE64_B64),
    DS_M0(DS_WRITE2_B32, DS_STORE_2ADDR_B32),
    DS_M0(DS_WRITE2_B64, DS_STORE_2ADDR_B64),
    DS_M0(DS_WRITE2ST64_B32, DS_STORE_2ADDR_STRIDE64_B32),
    DS_M0(DS_WRITE2ST64_B64, DS_STORE_2ADDR_STRIDE64_B64),
    DS_M0(DS_ADD_U32, DS_ADD_U32),
    DS_M0(DS_ADD_RTN_U32, DS_ADD_RTN_U32),
#define COMPARE_VARIANTS(Name, Suffix, Canonical) \
    E(V_CMP_##Name##Suffix##_e64, Canonical), \
    E(V_CMPX_##Name##Suffix##_e64, Canonical), \
    E(V_CMPX_##Name##Suffix##_nosdst_e64, Canonical),
#define VECTOR_COMPARE(Name, Predicate, BitWidth, SignExtendLiteral) \
    COMPARE_VARIANTS(Name, , V_CMP_##Name)
#define VECTOR_COMPARE16(Name, Predicate) \
    VECTOR_COMPARE(Name, Predicate, 16, false) \
    COMPARE_VARIANTS(Name, _t16, V_CMP_##Name) \
    COMPARE_VARIANTS(Name, _fake16, V_CMP_##Name)
#define VECTOR_CLASS(Name, BitWidth) \
    COMPARE_VARIANTS(CLASS_##Name, , V_CMP_CLASS_##Name)
#include "transpiler/decoder/vector-compare.def"
#undef COMPARE_VARIANTS
    E(V_ADD_F32_e64, V_ADD_F32),
    E(V_MUL_F32_e64, V_MUL_F32),
    E(V_SUB_F32_e64, V_SUB_F32),
    E(V_SUBREV_F32_e64, V_SUBREV_F32),
    E(V_FMAC_F32_e64, V_FMAC_F32),
    E(V_FMA_F32_e64, V_FMA_F32),
    E(V_FMAMK_F32, V_FMAMK_F32),
    E(V_FMAAK_F32, V_FMAAK_F32),
    E(V_MAX_F32_e64, V_MAX_NUM_F32),
    E(V_MIN_F32_e64, V_MIN_NUM_F32),
    E(V_ADD_F64_e64, V_ADD_F64),
    E(V_ADD_F64_pseudo_e64, V_ADD_F64),
    E(V_MUL_F64_e64, V_MUL_F64),
    E(V_MUL_F64_pseudo_e64, V_MUL_F64),
    E(V_FMA_F64_e64, V_FMA_F64),
    E(V_MAX_NUM_F64_e64, V_MAX_NUM_F64),
    E(V_MIN_NUM_F64_e64, V_MIN_NUM_F64),
    E(V_NOP_e64, V_NOP),
    E(V_FMAC_F64_e64, V_FMAC_F64),
    E(V_FMAMK_F64, V_FMAMK_F64),
    E(V_FMAMK_F64_gfx1250, V_FMAMK_F64),
    E(V_MOV_B32_e64, V_MOV_B32),
    E(V_MOV_B64_e64, V_MOV_B64),
    E(V_NOT_B32_e64, V_NOT_B32),
    E(V_BFREV_B32_e64, V_BFREV_B32),
    E(V_FFBH_U32_e64, V_FFBH_U32),
    E(V_FFBL_B32_e64, V_FFBL_B32),
    E(V_FFBH_I32_e64, V_FFBH_I32),
    E(V_CVT_F32_I32_e64, V_CVT_F32_I32),
    E(V_CVT_F32_U32_e64, V_CVT_F32_U32),
    E(V_CVT_I32_F32_e64, V_CVT_I32_F32),
    E(V_CVT_U32_F32_e64, V_CVT_U32_F32),
    E(V_CVT_F16_F32_e64, V_CVT_F16_F32),
    E(V_CVT_F16_F32_fake16_e64, V_CVT_F16_F32),
    E(V_CVT_F16_F32_t16_e64, V_CVT_F16_F32),
    E(V_CVT_F32_F16_e64, V_CVT_F32_F16),
    E(V_CVT_F32_F16_fake16_e64, V_CVT_F32_F16),
    E(V_CVT_F32_F16_t16_e64, V_CVT_F32_F16),
    E(V_CVT_F32_F64_e64, V_CVT_F32_F64),
    E(V_CVT_F64_F32_e64, V_CVT_F64_F32),
    E(V_CVT_F64_I32_e64, V_CVT_F64_I32),
    E(V_CVT_F64_U32_e64, V_CVT_F64_U32),
    E(V_CVT_I32_F64_e64, V_CVT_I32_F64),
    E(V_CVT_U32_F64_e64, V_CVT_U32_F64),
    E(V_TRUNC_F64_e64, V_TRUNC_F64),
    E(V_CEIL_F64_e64, V_CEIL_F64),
    E(V_RNDNE_F64_e64, V_RNDNE_F64),
    E(V_FLOOR_F64_e64, V_FLOOR_F64),
    E(V_RCP_F64_e64, V_RCP_F64),
    E(V_RSQ_F64_e64, V_RSQ_F64),
    E(V_CVT_F32_UBYTE0_e64, V_CVT_F32_UBYTE0),
    E(V_CVT_F32_UBYTE1_e64, V_CVT_F32_UBYTE1),
    E(V_CVT_F32_UBYTE2_e64, V_CVT_F32_UBYTE2),
    E(V_CVT_F32_UBYTE3_e64, V_CVT_F32_UBYTE3),
    E(V_FRACT_F32_e64, V_FRACT_F32),
    E(V_TRUNC_F32_e64, V_TRUNC_F32),
    E(V_CEIL_F32_e64, V_CEIL_F32),
    E(V_RNDNE_F32_e64, V_RNDNE_F32),
    E(V_FLOOR_F32_e64, V_FLOOR_F32),
    E(V_EXP_F32_e64, V_EXP_F32),
    E(V_LOG_F32_e64, V_LOG_F32),
    E(V_RCP_F32_e64, V_RCP_F32),
    E(V_RCP_IFLAG_F32_e64, V_RCP_IFLAG_F32),
    E(V_TANH_F32_e64, V_TANH_F32),
    E(V_S_EXP_F32_e64, V_S_EXP_F32),
    E(V_S_LOG_F32_e64, V_S_LOG_F32),
    E(V_S_RCP_F32_e64, V_S_RCP_F32),
    E(V_S_RSQ_F32_e64, V_S_RSQ_F32),
    E(V_S_SQRT_F32_e64, V_S_SQRT_F32),
    E(V_RSQ_F32_e64, V_RSQ_F32),
    E(V_SQRT_F32_e64, V_SQRT_F32),
    E(V_SIN_F32_e64, V_SIN_F32),
    E(V_COS_F32_e64, V_COS_F32),
    E(V_FREXP_EXP_I32_F32_e64, V_FREXP_EXP_I32_F32),
    E(V_FREXP_MANT_F32_e64, V_FREXP_MANT_F32),
    E(V_READFIRSTLANE_B32, V_READFIRSTLANE_B32),
    // gfx10 renamed the assembly mnemonics but retained these pseudos.
    E(V_ADD_U32_e64, V_ADD_NC_U32),
    E(V_SUB_U32_e64, V_SUB_NC_U32),
    E(V_SUBREV_U32_e64, V_SUBREV_NC_U32),
    E(V_ADD_CO_U32_e64, V_ADD_CO_U32),
    E(V_SUB_CO_U32_e64, V_SUB_CO_U32),
    E(V_SUBREV_CO_U32_e64, V_SUBREV_CO_U32),
    E(V_ADDC_U32_e64, V_ADD_CO_CI_U32),
    E(V_SUBB_U32_e64, V_SUB_CO_CI_U32),
    E(V_SUBBREV_U32_e64, V_SUBREV_CO_CI_U32),
    E(V_CNDMASK_B32_e64, V_CNDMASK_B32),
    E(V_MUL_I32_I24_e64, V_MUL_I32_I24),
    E(V_MUL_HI_I32_I24_e64, V_MUL_HI_I32_I24),
    E(V_MUL_U32_U24_e64, V_MUL_U32_U24),
    E(V_MUL_HI_U32_U24_e64, V_MUL_HI_U32_U24),
    E(V_MIN_I32_e64, V_MIN_I32),
    E(V_MAX_I32_e64, V_MAX_I32),
    E(V_MIN_U32_e64, V_MIN_U32),
    E(V_MAX_U32_e64, V_MAX_U32),
    E(V_AND_B32_e64, V_AND_B32),
    E(V_OR_B32_e64, V_OR_B32),
    E(V_XOR_B32_e64, V_XOR_B32),
    E(V_XNOR_B32_e64, V_XNOR_B32),
    E(V_BFM_B32_e64, V_BFM_B32),
    E(V_BCNT_U32_B32_e64, V_BCNT_U32_B32),
    E(V_LSHLREV_B32_e64, V_LSHLREV_B32),
    E(V_LSHRREV_B32_e64, V_LSHRREV_B32),
    E(V_ASHRREV_I32_e64, V_ASHRREV_I32),
    E(V_ADD_U64_e64, V_ADD_NC_U64),
    E(V_SUB_U64_e64, V_SUB_NC_U64),
    E(V_MUL_U64_e64, V_MUL_U64),
    E(V_LSHLREV_B64_pseudo_e64, V_LSHLREV_B64),
    // 16-bit integer operations. GFX11 gained true16 forms that name register
    // halves directly; the disassembler selects those over the whole-register
    // forms, so only the true16 spelling reaches this map on a GFX11+ source.
    E(V_ADD_U16_e64, V_ADD_U16),
    E(V_ADD_NC_U16_e64, V_ADD_U16),
    E(V_ADD_NC_U16_t16_e64, V_ADD_U16),
    E(V_ADD_I16_e64, V_ADD_I16),
    E(V_ADD_I16_t16_e64, V_ADD_I16),
    E(V_SUB_U16_e64, V_SUB_U16),
    E(V_SUB_NC_U16_e64, V_SUB_U16),
    E(V_SUB_NC_U16_t16_e64, V_SUB_U16),
    E(V_SUB_I16_e64, V_SUB_I16),
    E(V_SUB_I16_t16_e64, V_SUB_I16),
    E(V_SUBREV_U16_e64, V_SUBREV_U16),
    E(V_MUL_LO_U16_e64, V_MUL_LO_U16),
    E(V_MUL_LO_U16_t16_e64, V_MUL_LO_U16),
    // Only the GFX9 multiply-add pseudos are mapped: the older ones zero the
    // unwritten destination half instead of preserving it.
    E(V_MAD_U16_gfx9_e64, V_MAD_U16),
    E(V_MAD_U16_gfx9_t16_e64, V_MAD_U16),
    E(V_MAD_I16_gfx9_e64, V_MAD_I16),
    E(V_MAD_I16_gfx9_t16_e64, V_MAD_I16),
    E(V_LSHLREV_B16_e64, V_LSHLREV_B16),
    E(V_LSHLREV_B16_t16_e64, V_LSHLREV_B16),
    E(V_LSHRREV_B16_e64, V_LSHRREV_B16),
    E(V_LSHRREV_B16_t16_e64, V_LSHRREV_B16),
    E(V_ASHRREV_I16_e64, V_ASHRREV_I16),
    E(V_ASHRREV_I16_t16_e64, V_ASHRREV_I16),
    E(V_MIN_I16_e64, V_MIN_I16),
    E(V_MIN_I16_t16_e64, V_MIN_I16),
    E(V_MAX_I16_e64, V_MAX_I16),
    E(V_MAX_I16_t16_e64, V_MAX_I16),
    E(V_MIN_U16_e64, V_MIN_U16),
    E(V_MIN_U16_t16_e64, V_MIN_U16),
    E(V_MAX_U16_e64, V_MAX_U16),
    E(V_MAX_U16_t16_e64, V_MAX_U16),
    E(V_MIN3_I16_e64, V_MIN3_I16),
    E(V_MIN3_I16_t16_e64, V_MIN3_I16),
    E(V_MAX3_I16_e64, V_MAX3_I16),
    E(V_MAX3_I16_t16_e64, V_MAX3_I16),
    E(V_MIN3_U16_e64, V_MIN3_U16),
    E(V_MIN3_U16_t16_e64, V_MIN3_U16),
    E(V_MAX3_U16_e64, V_MAX3_U16),
    E(V_MAX3_U16_t16_e64, V_MAX3_U16),
    E(V_MED3_I16_e64, V_MED3_I16),
    E(V_MED3_I16_t16_e64, V_MED3_I16),
    E(V_MED3_U16_e64, V_MED3_U16),
    E(V_MED3_U16_t16_e64, V_MED3_U16),
    E(V_AND_B16_t16_e64, V_AND_B16),
    E(V_OR_B16_t16_e64, V_OR_B16),
    E(V_XOR_B16_t16_e64, V_XOR_B16),
    E(V_NOT_B16_t16_e64, V_NOT_B16),
    E(V_BITOP3_B16_e64, V_BITOP3_B16),
    E(V_BITOP3_B16_gfx1250_t16_e64, V_BITOP3_B16),
    E(V_MOV_B16_t16_e64, V_MOV_B16),
    E(V_CNDMASK_B16_t16_e64, V_CNDMASK_B16),
    E(V_DOT2C_I32_I16_e64, V_DOT2C_I32_I16),
    E(V_DOT4C_I32_I8_e64, V_DOT4C_I32_I8),
    E(V_DOT8C_I32_I4_e64, V_DOT8C_I32_I4),
    E(V_LDEXP_F32_e64, V_LDEXP_F32),
    E(V_PK_ADD_F16, V_PK_ADD_F16),
    E(V_PK_MUL_F16, V_PK_MUL_F16),
    E(V_PK_ADD_F32, V_PK_ADD_F32),
    E(V_PK_MUL_F32, V_PK_MUL_F32),
    E(V_PK_FMA_F32, V_PK_FMA_F32),
    E(V_PK_FMA_F32_gfx1250, V_PK_FMA_F32),
    E(V_FMA_MIX_F32_BF16, V_FMA_MIX_F32_BF16),
    E(V_FMA_MIX_F32_BF16_gfx1250, V_FMA_MIX_F32_BF16),
    E(V_PK_ADD_F32_gfx1250, V_PK_ADD_F32),
    E(V_PK_MUL_F32_gfx1250, V_PK_MUL_F32),
    E(V_PK_ADD_U16, V_PK_ADD_U16),
    E(V_PK_ADD_I16, V_PK_ADD_I16),
    E(V_PK_SUB_U16, V_PK_SUB_U16),
    E(V_PK_SUB_I16, V_PK_SUB_I16),
    E(V_PK_MUL_LO_U16, V_PK_MUL_LO_U16),
    E(V_PK_MAD_U16, V_PK_MAD_U16),
    E(V_PK_MAD_I16, V_PK_MAD_I16),
    E(V_PK_LSHLREV_B16, V_PK_LSHLREV_B16),
    E(V_PK_LSHRREV_B16, V_PK_LSHRREV_B16),
    E(V_PK_ASHRREV_I16, V_PK_ASHRREV_I16),
    E(V_PK_MIN_I16, V_PK_MIN_I16),
    E(V_PK_MAX_I16, V_PK_MAX_I16),
    E(V_PK_MIN_U16, V_PK_MIN_U16),
    E(V_PK_MAX_U16, V_PK_MAX_U16),
    E(V_PK_MIN3_I16, V_PK_MIN3_I16),
    E(V_PK_MAX3_I16, V_PK_MAX3_I16),
    E(V_PK_MIN3_U16, V_PK_MIN3_U16),
    E(V_PK_MAX3_U16, V_PK_MAX3_U16),
    E(V_DIV_SCALE_F32_e64, V_DIV_SCALE_F32),
    E(V_LDEXP_F64_e64, V_LDEXP_F64),
    E(V_DIV_FMAS_F32_e64, V_DIV_FMAS_F32),
    E(V_DIV_FIXUP_F32_e64, V_DIV_FIXUP_F32),
    E(V_MAX3_F32_e64, V_MAX3_NUM_F32),
    E(V_MIN3_F32_e64, V_MIN3_NUM_F32),
    E(V_MED3_F32_e64, V_MED3_NUM_F32),
    E(V_MAXIMUM3_F32_e64, V_MAXIMUM3_F32),
    E(V_MINIMUM3_F32_e64, V_MINIMUM3_F32),
    E(V_MAXMIN_F32_e64, V_MAXMIN_NUM_F32),
    E(V_MINMAX_F32_e64, V_MINMAX_NUM_F32),
    E(V_MAXIMUMMINIMUM_F32_e64, V_MAXIMUMMINIMUM_F32),
    E(V_MINIMUMMAXIMUM_F32_e64, V_MINIMUMMAXIMUM_F32),
    E(V_MAXIMUM_F32_e64, V_MAXIMUM_F32),
    E(V_MINIMUM_F32_e64, V_MINIMUM_F32),
    E(V_WMMA_F32_16X16X4_F32_w32_twoaddr, V_WMMA_F32_16x16x4_F32),
    E(V_WMMA_F32_16X16X4_F32_w32_threeaddr, V_WMMA_F32_16x16x4_F32),
    E(V_WMMA_F32_16X16X32_F16_w32_twoaddr, V_WMMA_F32_16x16x32_F16),
    E(V_WMMA_F32_16X16X32_F16_w32_threeaddr, V_WMMA_F32_16x16x32_F16),
    E(V_WMMA_F32_16X16X32_BF16_w32_twoaddr, V_WMMA_F32_16x16x32_BF16),
    E(V_WMMA_F32_16X16X32_BF16_w32_threeaddr, V_WMMA_F32_16x16x32_BF16),
    E(V_WMMA_I32_16X16X64_IU8_w32_twoaddr, V_WMMA_I32_16x16x64_IU8),
    E(V_WMMA_I32_16X16X64_IU8_w32_threeaddr, V_WMMA_I32_16x16x64_IU8),
    E(V_MFMA_F32_16X16X16F16_e64, V_MFMA_F32_16x16x16_F16),
    E(V_MFMA_F32_16X16X16F16_vgprcd_e64, V_MFMA_F32_16x16x16_F16),
    E(V_MFMA_F32_16X16X16BF16_1K_e64, V_MFMA_F32_16x16x16_BF16_1K),
    E(V_MFMA_F32_16X16X16BF16_1K_vgprcd_e64,
      V_MFMA_F32_16x16x16_BF16_1K),
    E(V_MFMA_I32_16X16X32I8_e64, V_MFMA_I32_16x16x32_I8),
    E(V_MFMA_I32_16X16X32I8_vgprcd_e64, V_MFMA_I32_16x16x32_I8),
    E(V_ADD_I32_e64, V_ADD_I32),
    E(V_SUB_I32_e64, V_SUB_I32),
    E(V_MUL_LO_U32_e64, V_MUL_LO_U32),
    E(V_MUL_HI_U32_e64, V_MUL_HI_U32),
    E(V_MUL_HI_I32_e64, V_MUL_HI_I32),
    E(V_MAD_I32_I24_e64, V_MAD_I32_I24),
    E(V_MAD_U32_U24_e64, V_MAD_U32_U24),
    E(V_MAD_U32_e64, V_MAD_U32),
    E(V_ADD3_U32_e64, V_ADD3_U32),
    E(V_ADD_MIN_I32_e64, V_ADD_MIN_I32),
    E(V_ADD_MAX_I32_e64, V_ADD_MAX_I32),
    E(V_ADD_MIN_U32_e64, V_ADD_MIN_U32),
    E(V_ADD_MAX_U32_e64, V_ADD_MAX_U32),
    E(V_MIN3_I32_e64, V_MIN3_I32),
    E(V_MAX3_I32_e64, V_MAX3_I32),
    E(V_MIN3_U32_e64, V_MIN3_U32),
    E(V_MAX3_U32_e64, V_MAX3_U32),
    E(V_MED3_I32_e64, V_MED3_I32),
    E(V_MINMAX_I32_e64, V_MINMAX_I32),
    E(V_MAXMIN_I32_e64, V_MAXMIN_I32),
    E(V_MINMAX_U32_e64, V_MINMAX_U32),
    E(V_MAXMIN_U32_e64, V_MAXMIN_U32),
    E(V_MIN_I64_e64, V_MIN_I64),
    E(V_MAX_I64_e64, V_MAX_I64),
    E(V_MIN_U64_e64, V_MIN_U64),
    E(V_MAX_U64_e64, V_MAX_U64),
    E(V_MAD_U64_U32_e64, V_MAD_U64_U32),
    E(V_MAD_I64_I32_e64, V_MAD_I64_I32),
    E(V_MAD_NC_U64_U32_e64, V_MAD_NC_U64_U32),
    E(V_MAD_NC_I64_I32_e64, V_MAD_NC_I64_I32),
    E(V_LSHL_ADD_U32_e64, V_LSHL_ADD_U32),
    E(V_ADD_LSHL_U32_e64, V_ADD_LSHL_U32),
    E(V_LSHL_OR_B32_e64, V_LSHL_OR_B32),
    E(V_AND_OR_B32_e64, V_AND_OR_B32),
    E(V_OR3_B32_e64, V_OR3_B32),
    E(V_XOR3_B32_e64, V_XOR3_B32),
    E(V_XAD_U32_e64, V_XAD_U32),
    E(V_ALIGNBIT_B32_e64, V_ALIGNBIT_B32),
    E(V_ALIGNBIT_B32_fake16_e64, V_ALIGNBIT_B32),
    E(V_ALIGNBIT_B32_opsel_e64, V_ALIGNBIT_B32),
    E(V_ALIGNBIT_B32_t16_e64, V_ALIGNBIT_B32),
    E(V_BFE_U32_e64, V_BFE_U32),
    E(V_BFE_I32_e64, V_BFE_I32),
    E(V_BFI_B32_e64, V_BFI_B32),
    E(V_BITOP3_B32_e64, V_BITOP3_B32),
    E(V_PERM_B32_e64, V_PERM_B32),
    E(V_MBCNT_LO_U32_B32_e64, V_MBCNT_LO_U32_B32),
    E(V_MBCNT_HI_U32_B32_e64, V_MBCNT_HI_U32_B32),
    E(V_READLANE_B32, V_READLANE_B32),
    E(V_WRITELANE_B32, V_WRITELANE_B32),
    E(V_LSHRREV_B64_e64, V_LSHRREV_B64),
    E(V_ASHRREV_I64_e64, V_ASHRREV_I64),
    E(V_LSHL_ADD_U64_e64, V_LSHL_ADD_U64),
    // clang-format on
};

#undef E
#undef BUFFER_RAW
#undef DS_M0

// Update this bound when SIEncodingFamily gains a new value, otherwise opcodes
// using that encoding remain unmapped.
constexpr unsigned KNumEncodingFamilies =
    static_cast<unsigned>(SIEncodingFamily::GFX13) + 1;

// Reverse map MC opcode -> canonical pseudo, built by scanning the first
// `NumOpc` pseudos across every encoding family.
DenseMap<unsigned, unsigned> buildMcToPseudoMap(unsigned NumOpc) {
  DenseMap<unsigned, unsigned> Result;
  for (unsigned P = 0; P < NumOpc; ++P) {
    for (unsigned Gen = 0; Gen < KNumEncodingFamilies; ++Gen) {
      std::optional<unsigned> Mc =
          mappedOpcode(transpiler::getMCOpcode(P, Gen));
      if (Mc && *Mc != P)
        Result.try_emplace(*Mc, P);
    }
  }
  return Result;
}

// Reverse map DPP opcode -> base VOP opcode, built by scanning the first
// `NumOpc` opcodes because only the forward mappings are exposed.
DenseMap<unsigned, unsigned> buildDppToBaseMap(unsigned NumOpc) {
  DenseMap<unsigned, unsigned> Result;
  for (unsigned P = 0; P < NumOpc; ++P) {
    if (std::optional<unsigned> D32 = mappedOpcode(transpiler::getDPPOp32(P)))
      Result.try_emplace(*D32, P);
    if (std::optional<unsigned> D64 = mappedOpcode(transpiler::getDPPOp64(P)))
      Result.try_emplace(*D64, P);
  }
  return Result;
}

// Map `Mc` to the canonical pseudo used by kCanonTable.
unsigned canonicalize(unsigned Mc, const MCInstrInfo &MCII,
                      const DenseMap<unsigned, unsigned> &McToPseudo,
                      const DenseMap<unsigned, unsigned> &DppToBase) {
  unsigned P = Mc;

  DenseMap<unsigned, unsigned>::const_iterator PseudoIt = McToPseudo.find(P);
  if (PseudoIt != McToPseudo.end())
    P = PseudoIt->second;

  DenseMap<unsigned, unsigned>::const_iterator DppIt = DppToBase.find(P);
  if (DppIt != DppToBase.end())
    P = DppIt->second;

  if (std::optional<unsigned> Base =
          mappedOpcode(transpiler::getBasicFromSDWAOp(P)))
    P = *Base;

  if (std::optional<unsigned> E64 = mappedOpcode(transpiler::getVOPe64(P)))
    P = *E64;

  // Testing the format flag first avoids a table lookup for every non-FLAT
  // opcode.
  if (P < MCII.getNumOpcodes() && SIInstrFlags::isFLAT(MCII, P)) {
    if (std::optional<unsigned> Vaddr =
            mappedOpcode(transpiler::getGlobalVaddrOp(P)))
      P = *Vaddr;
  }

  return P;
}

} // namespace

CanonicalOp OpcodeMap::lookup(unsigned Opcode) const {
  DenseMap<unsigned, CanonicalOp>::const_iterator It = Map.find(Opcode);
  return It != Map.end() ? It->second : CanonicalOp::Unknown;
}

void OpcodeMap::build(const MCInstrInfo &MCII) {
  // A duplicate opcode would silently keep only the first row and route the
  // rest through the wrong CanonicalOp, so it is a table-authoring bug.
  DenseMap<unsigned, CanonicalOp> CanonToSem;
  CanonToSem.reserve(std::size(kCanonTable));
  for (const Entry &E : kCanonTable) {
    bool Inserted = CanonToSem.try_emplace(E.Opc, E.Sem).second;
    assert(Inserted && "kCanonTable maps one MC opcode to two CanonicalOps");
    (void)Inserted;
  }

  const unsigned NumOpc = MCII.getNumOpcodes();
  const DenseMap<unsigned, unsigned> McToPseudo = buildMcToPseudoMap(NumOpc);
  const DenseMap<unsigned, unsigned> DppToBase = buildDppToBaseMap(NumOpc);

  Map.clear();
  for (unsigned Mc = 0; Mc < NumOpc; ++Mc) {
    const unsigned Canon = canonicalize(Mc, MCII, McToPseudo, DppToBase);
    DenseMap<unsigned, CanonicalOp>::const_iterator It = CanonToSem.find(Canon);
    if (It != CanonToSem.end())
      Map[Mc] = It->second;
  }
}

} // namespace COMGR::transpiler
