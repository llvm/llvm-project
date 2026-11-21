//===- RISCVCPUFeatures.inc - RISC-V __riscv_feature_bits layout -*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Group ID and bitmask of each extension in __riscv_feature_bits.
//
// NOTE: Should sync-up with RISCVFeatures.td
// TODO: Maybe generate a header from tablegen then include it.
#define A_GROUPID 0
#define A_BITMASK (1ULL << 0)
#define B_GROUPID 0
#define B_BITMASK (1ULL << 1)
#define C_GROUPID 0
#define C_BITMASK (1ULL << 2)
#define D_GROUPID 0
#define D_BITMASK (1ULL << 3)
#define E_GROUPID 0
#define E_BITMASK (1ULL << 4)
#define F_GROUPID 0
#define F_BITMASK (1ULL << 5)
#define H_GROUPID 0
#define H_BITMASK (1ULL << 7)
#define I_GROUPID 0
#define I_BITMASK (1ULL << 8)
#define M_GROUPID 0
#define M_BITMASK (1ULL << 12)
#define Q_GROUPID 0
#define Q_BITMASK (1ULL << 16)
#define V_GROUPID 0
#define V_BITMASK (1ULL << 21)
#define ZACAS_GROUPID 0
#define ZACAS_BITMASK (1ULL << 26)
#define ZBA_GROUPID 0
#define ZBA_BITMASK (1ULL << 27)
#define ZBB_GROUPID 0
#define ZBB_BITMASK (1ULL << 28)
#define ZBC_GROUPID 0
#define ZBC_BITMASK (1ULL << 29)
#define ZBKB_GROUPID 0
#define ZBKB_BITMASK (1ULL << 30)
#define ZBKC_GROUPID 0
#define ZBKC_BITMASK (1ULL << 31)
#define ZBKX_GROUPID 0
#define ZBKX_BITMASK (1ULL << 32)
#define ZBS_GROUPID 0
#define ZBS_BITMASK (1ULL << 33)
#define ZFA_GROUPID 0
#define ZFA_BITMASK (1ULL << 34)
#define ZFH_GROUPID 0
#define ZFH_BITMASK (1ULL << 35)
#define ZFHMIN_GROUPID 0
#define ZFHMIN_BITMASK (1ULL << 36)
#define ZICBOZ_GROUPID 0
#define ZICBOZ_BITMASK (1ULL << 37)
#define ZICOND_GROUPID 0
#define ZICOND_BITMASK (1ULL << 38)
#define ZIHINTNTL_GROUPID 0
#define ZIHINTNTL_BITMASK (1ULL << 39)
#define ZIHINTPAUSE_GROUPID 0
#define ZIHINTPAUSE_BITMASK (1ULL << 40)
#define ZKND_GROUPID 0
#define ZKND_BITMASK (1ULL << 41)
#define ZKNE_GROUPID 0
#define ZKNE_BITMASK (1ULL << 42)
#define ZKNH_GROUPID 0
#define ZKNH_BITMASK (1ULL << 43)
#define ZKSED_GROUPID 0
#define ZKSED_BITMASK (1ULL << 44)
#define ZKSH_GROUPID 0
#define ZKSH_BITMASK (1ULL << 45)
#define ZKT_GROUPID 0
#define ZKT_BITMASK (1ULL << 46)
#define ZTSO_GROUPID 0
#define ZTSO_BITMASK (1ULL << 47)
#define ZVBB_GROUPID 0
#define ZVBB_BITMASK (1ULL << 48)
#define ZVBC_GROUPID 0
#define ZVBC_BITMASK (1ULL << 49)
#define ZVFH_GROUPID 0
#define ZVFH_BITMASK (1ULL << 50)
#define ZVFHMIN_GROUPID 0
#define ZVFHMIN_BITMASK (1ULL << 51)
#define ZVKB_GROUPID 0
#define ZVKB_BITMASK (1ULL << 52)
#define ZVKG_GROUPID 0
#define ZVKG_BITMASK (1ULL << 53)
#define ZVKNED_GROUPID 0
#define ZVKNED_BITMASK (1ULL << 54)
#define ZVKNHA_GROUPID 0
#define ZVKNHA_BITMASK (1ULL << 55)
#define ZVKNHB_GROUPID 0
#define ZVKNHB_BITMASK (1ULL << 56)
#define ZVKSED_GROUPID 0
#define ZVKSED_BITMASK (1ULL << 57)
#define ZVKSH_GROUPID 0
#define ZVKSH_BITMASK (1ULL << 58)
#define ZVKT_GROUPID 0
#define ZVKT_BITMASK (1ULL << 59)
#define ZVE32X_GROUPID 0
#define ZVE32X_BITMASK (1ULL << 60)
#define ZVE32F_GROUPID 0
#define ZVE32F_BITMASK (1ULL << 61)
#define ZVE64X_GROUPID 0
#define ZVE64X_BITMASK (1ULL << 62)
#define ZVE64F_GROUPID 0
#define ZVE64F_BITMASK (1ULL << 63)
#define ZVE64D_GROUPID 1
#define ZVE64D_BITMASK (1ULL << 0)
#define ZIMOP_GROUPID 1
#define ZIMOP_BITMASK (1ULL << 1)
#define ZCA_GROUPID 1
#define ZCA_BITMASK (1ULL << 2)
#define ZCB_GROUPID 1
#define ZCB_BITMASK (1ULL << 3)
#define ZCD_GROUPID 1
#define ZCD_BITMASK (1ULL << 4)
#define ZCF_GROUPID 1
#define ZCF_BITMASK (1ULL << 5)
#define ZCMOP_GROUPID 1
#define ZCMOP_BITMASK (1ULL << 6)
#define ZAWRS_GROUPID 1
#define ZAWRS_BITMASK (1ULL << 7)
#define ZILSD_GROUPID 1
#define ZILSD_BITMASK (1ULL << 8)
#define ZCLSD_GROUPID 1
#define ZCLSD_BITMASK (1ULL << 9)
#define ZCMP_GROUPID 1
#define ZCMP_BITMASK (1ULL << 10)
#define ZIFENCEI_GROUPID 1
#define ZIFENCEI_BITMASK (1ULL << 11)
#define ZMMUL_GROUPID 1
#define ZMMUL_BITMASK (1ULL << 12)
#define ZIBI_GROUPID 1
#define ZIBI_BITMASK (1ULL << 13)
#define SUPM_GROUPID 1
#define SUPM_BITMASK (1ULL << 14)
#define ZICNTR_GROUPID 1
#define ZICNTR_BITMASK (1ULL << 15)
#define ZIHPM_GROUPID 1
#define ZIHPM_BITMASK (1ULL << 16)
#define ZFBFMIN_GROUPID 1
#define ZFBFMIN_BITMASK (1ULL << 17)
#define ZVFBFMIN_GROUPID 1
#define ZVFBFMIN_BITMASK (1ULL << 18)
#define ZVFBFWMA_GROUPID 1
#define ZVFBFWMA_BITMASK (1ULL << 19)
#define ZICBOM_GROUPID 1
#define ZICBOM_BITMASK (1ULL << 20)
#define ZAAMO_GROUPID 1
#define ZAAMO_BITMASK (1ULL << 21)
#define ZALRSC_GROUPID 1
#define ZALRSC_BITMASK (1ULL << 22)
#define ZABHA_GROUPID 1
#define ZABHA_BITMASK (1ULL << 23)
#define ZALASR_GROUPID 1
#define ZALASR_BITMASK (1ULL << 24)
#define ZICBOP_GROUPID 1
#define ZICBOP_BITMASK (1ULL << 25)
#define ZICFISS_GROUPID 1
#define ZICFISS_BITMASK (1ULL << 27)
