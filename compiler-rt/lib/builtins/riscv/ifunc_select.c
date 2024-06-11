//=== ifunc_select.c - Check environment hardware feature -*- C -*-===========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

static long syscall_impl_5_args(long number, long arg1, long arg2, long arg3,
                                long arg4, long arg5) {
  register long a7 __asm__("a7") = number;
  register long a0 __asm__("a0") = arg1;
  register long a1 __asm__("a1") = arg2;
  register long a2 __asm__("a2") = arg3;
  register long a3 __asm__("a3") = arg4;
  register long a4 __asm__("a4") = arg5;
  __asm__ __volatile__("ecall\n\t"
                       : "=r"(a0)
                       : "r"(a7), "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4)
                       : "memory");
  return a0;
}

#define RISCV_HWPROBE_KEY_MVENDORID 0
#define RISCV_HWPROBE_KEY_MARCHID 1
#define RISCV_HWPROBE_KEY_MIMPID 2
#define RISCV_HWPROBE_KEY_BASE_BEHAVIOR 3
#define RISCV_HWPROBE_BASE_BEHAVIOR_IMA (1ULL << 0)
#define RISCV_HWPROBE_KEY_IMA_EXT_0 4
#define RISCV_HWPROBE_IMA_FD (1ULL << 0)
#define RISCV_HWPROBE_IMA_C (1ULL << 1)
#define RISCV_HWPROBE_IMA_V (1ULL << 2)
#define RISCV_HWPROBE_EXT_ZBA (1ULL << 3)
#define RISCV_HWPROBE_EXT_ZBB (1ULL << 4)
#define RISCV_HWPROBE_EXT_ZBS (1ULL << 5)
#define RISCV_HWPROBE_EXT_ZICBOZ (1ULL << 6)
#define RISCV_HWPROBE_EXT_ZBC (1ULL << 7)
#define RISCV_HWPROBE_EXT_ZBKB (1ULL << 8)
#define RISCV_HWPROBE_EXT_ZBKC (1ULL << 9)
#define RISCV_HWPROBE_EXT_ZBKX (1ULL << 10)
#define RISCV_HWPROBE_EXT_ZKND (1ULL << 11)
#define RISCV_HWPROBE_EXT_ZKNE (1ULL << 12)
#define RISCV_HWPROBE_EXT_ZKNH (1ULL << 13)
#define RISCV_HWPROBE_EXT_ZKSED (1ULL << 14)
#define RISCV_HWPROBE_EXT_ZKSH (1ULL << 15)
#define RISCV_HWPROBE_EXT_ZKT (1ULL << 16)
#define RISCV_HWPROBE_EXT_ZVBB (1ULL << 17)
#define RISCV_HWPROBE_EXT_ZVBC (1ULL << 18)
#define RISCV_HWPROBE_EXT_ZVKB (1ULL << 19)
#define RISCV_HWPROBE_EXT_ZVKG (1ULL << 20)
#define RISCV_HWPROBE_EXT_ZVKNED (1ULL << 21)
#define RISCV_HWPROBE_EXT_ZVKNHA (1ULL << 22)
#define RISCV_HWPROBE_EXT_ZVKNHB (1ULL << 23)
#define RISCV_HWPROBE_EXT_ZVKSED (1ULL << 24)
#define RISCV_HWPROBE_EXT_ZVKSH (1ULL << 25)
#define RISCV_HWPROBE_EXT_ZVKT (1ULL << 26)
#define RISCV_HWPROBE_EXT_ZFH (1ULL << 27)
#define RISCV_HWPROBE_EXT_ZFHMIN (1ULL << 28)
#define RISCV_HWPROBE_EXT_ZIHINTNTL (1ULL << 29)
#define RISCV_HWPROBE_EXT_ZVFH (1ULL << 30)
#define RISCV_HWPROBE_EXT_ZVFHMIN (1ULL << 31)
#define RISCV_HWPROBE_EXT_ZFA (1ULL << 32)
#define RISCV_HWPROBE_EXT_ZTSO (1ULL << 33)
#define RISCV_HWPROBE_EXT_ZACAS (1ULL << 34)
#define RISCV_HWPROBE_EXT_ZICOND (1ULL << 35)
#define RISCV_HWPROBE_KEY_CPUPERF_0 5
#define RISCV_HWPROBE_MISALIGNED_UNKNOWN (0 << 0)
#define RISCV_HWPROBE_MISALIGNED_EMULATED (1ULL << 0)
#define RISCV_HWPROBE_MISALIGNED_SLOW (2 << 0)
#define RISCV_HWPROBE_MISALIGNED_FAST (3 << 0)
#define RISCV_HWPROBE_MISALIGNED_UNSUPPORTED (4 << 0)
#define RISCV_HWPROBE_MISALIGNED_MASK (7 << 0)
#define RISCV_HWPROBE_KEY_ZICBOZ_BLOCK_SIZE 6
/* Increase RISCV_HWPROBE_MAX_KEY when adding items. */

/* Flags */
#define RISCV_HWPROBE_WHICH_CPUS (1ULL << 0)

struct riscv_hwprobe {
  long long key;
  unsigned long long value;
};

/* Size definition for CPU sets.  */
#define __CPU_SETSIZE 1024
#define __NCPUBITS (8 * sizeof(unsigned long int))

/* Data structure to describe CPU mask.  */
typedef struct {
  unsigned long int __bits[__CPU_SETSIZE / __NCPUBITS];
} cpu_set_t;

#define SYS_riscv_hwprobe 258
static long sys_riscv_hwprobe(struct riscv_hwprobe *pairs, unsigned pair_count,
                              unsigned cpu_count, cpu_set_t *cpus,
                              unsigned int flags) {
  return syscall_impl_5_args(SYS_riscv_hwprobe, (long)pairs, pair_count,
                             cpu_count, (long)cpus, flags);
}

static long initHwProbe(struct riscv_hwprobe *Hwprobes, int len) {
  return sys_riscv_hwprobe(Hwprobes, len, 0, (cpu_set_t *)((void *)0), 0);
}

struct {
  unsigned length;
  unsigned long long features[2];
} __riscv_feature_bits __attribute__((visibility("hidden"), nocommon));

struct {
  unsigned vendorID;
  unsigned length;
  unsigned long long features[1];
} __riscv_vendor_feature_bits __attribute__((visibility("hidden"), nocommon));

// NOTE: Should sync-up with RISCVFeatures.td
// TODO: Maybe generate a header from tablegen then include it.
#define A_GROUPID 0
#define A_BITMASK (1ULL << 22)
#define C_GROUPID 0
#define C_BITMASK (1ULL << 43)
#define D_GROUPID 0
#define D_BITMASK (1ULL << 34)
#define E_GROUPID 0
#define E_BITMASK (1ULL << 1)
#define F_GROUPID 0
#define F_BITMASK (1ULL << 33)
#define H_GROUPID 1
#define H_BITMASK (1ULL << 43)
#define I_GROUPID 0
#define I_BITMASK (1ULL << 0)
#define M_GROUPID 0
#define M_BITMASK (1ULL << 20)
#define V_GROUPID 1
#define V_BITMASK (1ULL << 22)
#define ZA128RS_GROUPID 0
#define ZA128RS_BITMASK (1ULL << 25)
#define ZA64RS_GROUPID 0
#define ZA64RS_BITMASK (1ULL << 24)
#define ZAAMO_GROUPID 0
#define ZAAMO_BITMASK (1ULL << 26)
#define ZABHA_GROUPID 0
#define ZABHA_BITMASK (1ULL << 27)
#define ZACAS_GROUPID 0
#define ZACAS_BITMASK (1ULL << 28)
#define ZALASR_GROUPID 0
#define ZALASR_BITMASK (1ULL << 29)
#define ZALRSC_GROUPID 0
#define ZALRSC_BITMASK (1ULL << 30)
#define ZAMA16B_GROUPID 0
#define ZAMA16B_BITMASK (1ULL << 31)
#define ZAWRS_GROUPID 0
#define ZAWRS_BITMASK (1ULL << 32)
#define ZBA_GROUPID 0
#define ZBA_BITMASK (1ULL << 52)
#define ZBB_GROUPID 0
#define ZBB_BITMASK (1ULL << 53)
#define ZBC_GROUPID 0
#define ZBC_BITMASK (1ULL << 54)
#define ZBKB_GROUPID 0
#define ZBKB_BITMASK (1ULL << 56)
#define ZBKC_GROUPID 0
#define ZBKC_BITMASK (1ULL << 58)
#define ZBKX_GROUPID 0
#define ZBKX_BITMASK (1ULL << 57)
#define ZBS_GROUPID 0
#define ZBS_BITMASK (1ULL << 55)
#define ZCA_GROUPID 0
#define ZCA_BITMASK (1ULL << 44)
#define ZCB_GROUPID 0
#define ZCB_BITMASK (1ULL << 45)
#define ZCD_GROUPID 0
#define ZCD_BITMASK (1ULL << 46)
#define ZCE_GROUPID 0
#define ZCE_BITMASK (1ULL << 50)
#define ZCF_GROUPID 0
#define ZCF_BITMASK (1ULL << 47)
#define ZCMOP_GROUPID 0
#define ZCMOP_BITMASK (1ULL << 51)
#define ZCMP_GROUPID 0
#define ZCMP_BITMASK (1ULL << 48)
#define ZCMT_GROUPID 0
#define ZCMT_BITMASK (1ULL << 49)
#define ZDINX_GROUPID 0
#define ZDINX_BITMASK (1ULL << 40)
#define ZFA_GROUPID 0
#define ZFA_BITMASK (1ULL << 38)
#define ZFBFMIN_GROUPID 0
#define ZFBFMIN_BITMASK (1ULL << 37)
#define ZFH_GROUPID 0
#define ZFH_BITMASK (1ULL << 36)
#define ZFHMIN_GROUPID 0
#define ZFHMIN_BITMASK (1ULL << 35)
#define ZFINX_GROUPID 0
#define ZFINX_BITMASK (1ULL << 39)
#define ZHINX_GROUPID 0
#define ZHINX_BITMASK (1ULL << 42)
#define ZHINXMIN_GROUPID 0
#define ZHINXMIN_BITMASK (1ULL << 41)
#define ZIC64B_GROUPID 0
#define ZIC64B_BITMASK (1ULL << 2)
#define ZICBOM_GROUPID 0
#define ZICBOM_BITMASK (1ULL << 3)
#define ZICBOP_GROUPID 0
#define ZICBOP_BITMASK (1ULL << 4)
#define ZICBOZ_GROUPID 0
#define ZICBOZ_BITMASK (1ULL << 5)
#define ZICCAMOA_GROUPID 0
#define ZICCAMOA_BITMASK (1ULL << 6)
#define ZICCIF_GROUPID 0
#define ZICCIF_BITMASK (1ULL << 7)
#define ZICCLSM_GROUPID 0
#define ZICCLSM_BITMASK (1ULL << 8)
#define ZICCRSE_GROUPID 0
#define ZICCRSE_BITMASK (1ULL << 9)
#define ZICFILP_GROUPID 0
#define ZICFILP_BITMASK (1ULL << 18)
#define ZICFISS_GROUPID 0
#define ZICFISS_BITMASK (1ULL << 19)
#define ZICNTR_GROUPID 0
#define ZICNTR_BITMASK (1ULL << 11)
#define ZICOND_GROUPID 0
#define ZICOND_BITMASK (1ULL << 12)
#define ZICSR_GROUPID 0
#define ZICSR_BITMASK (1ULL << 10)
#define ZIFENCEI_GROUPID 0
#define ZIFENCEI_BITMASK (1ULL << 13)
#define ZIHINTNTL_GROUPID 0
#define ZIHINTNTL_BITMASK (1ULL << 15)
#define ZIHINTPAUSE_GROUPID 0
#define ZIHINTPAUSE_BITMASK (1ULL << 14)
#define ZIHPM_GROUPID 0
#define ZIHPM_BITMASK (1ULL << 16)
#define ZIMOP_GROUPID 0
#define ZIMOP_BITMASK (1ULL << 17)
#define ZK_GROUPID 1
#define ZK_BITMASK (1ULL << 4)
#define ZKN_GROUPID 1
#define ZKN_BITMASK (1ULL << 1)
#define ZKND_GROUPID 0
#define ZKND_BITMASK (1ULL << 59)
#define ZKNE_GROUPID 0
#define ZKNE_BITMASK (1ULL << 60)
#define ZKNH_GROUPID 0
#define ZKNH_BITMASK (1ULL << 61)
#define ZKR_GROUPID 1
#define ZKR_BITMASK (1ULL << 0)
#define ZKS_GROUPID 1
#define ZKS_BITMASK (1ULL << 2)
#define ZKSED_GROUPID 0
#define ZKSED_BITMASK (1ULL << 62)
#define ZKSH_GROUPID 0
#define ZKSH_BITMASK (1ULL << 63)
#define ZKT_GROUPID 1
#define ZKT_BITMASK (1ULL << 3)
#define ZMMUL_GROUPID 0
#define ZMMUL_BITMASK (1ULL << 21)
#define ZTSO_GROUPID 0
#define ZTSO_BITMASK (1ULL << 23)
#define ZVBB_GROUPID 1
#define ZVBB_BITMASK (1ULL << 28)
#define ZVBC_GROUPID 1
#define ZVBC_BITMASK (1ULL << 29)
#define ZVE32F_GROUPID 1
#define ZVE32F_BITMASK (1ULL << 18)
#define ZVE32X_GROUPID 1
#define ZVE32X_BITMASK (1ULL << 17)
#define ZVE64D_GROUPID 1
#define ZVE64D_BITMASK (1ULL << 21)
#define ZVE64F_GROUPID 1
#define ZVE64F_BITMASK (1ULL << 20)
#define ZVE64X_GROUPID 1
#define ZVE64X_BITMASK (1ULL << 19)
#define ZVFBFMIN_GROUPID 1
#define ZVFBFMIN_BITMASK (1ULL << 23)
#define ZVFBFWMA_GROUPID 1
#define ZVFBFWMA_BITMASK (1ULL << 24)
#define ZVFH_GROUPID 1
#define ZVFH_BITMASK (1ULL << 26)
#define ZVFHMIN_GROUPID 1
#define ZVFHMIN_BITMASK (1ULL << 25)
#define ZVKB_GROUPID 1
#define ZVKB_BITMASK (1ULL << 27)
#define ZVKG_GROUPID 1
#define ZVKG_BITMASK (1ULL << 30)
#define ZVKN_GROUPID 1
#define ZVKN_BITMASK (1ULL << 37)
#define ZVKNC_GROUPID 1
#define ZVKNC_BITMASK (1ULL << 38)
#define ZVKNED_GROUPID 1
#define ZVKNED_BITMASK (1ULL << 31)
#define ZVKNG_GROUPID 1
#define ZVKNG_BITMASK (1ULL << 39)
#define ZVKNHA_GROUPID 1
#define ZVKNHA_BITMASK (1ULL << 32)
#define ZVKNHB_GROUPID 1
#define ZVKNHB_BITMASK (1ULL << 33)
#define ZVKS_GROUPID 1
#define ZVKS_BITMASK (1ULL << 40)
#define ZVKSC_GROUPID 1
#define ZVKSC_BITMASK (1ULL << 41)
#define ZVKSED_GROUPID 1
#define ZVKSED_BITMASK (1ULL << 34)
#define ZVKSG_GROUPID 1
#define ZVKSG_BITMASK (1ULL << 42)
#define ZVKSH_GROUPID 1
#define ZVKSH_BITMASK (1ULL << 35)
#define ZVKT_GROUPID 1
#define ZVKT_BITMASK (1ULL << 36)
#define ZVL1024B_GROUPID 1
#define ZVL1024B_BITMASK (1ULL << 10)
#define ZVL128B_GROUPID 1
#define ZVL128B_BITMASK (1ULL << 7)
#define ZVL16384B_GROUPID 1
#define ZVL16384B_BITMASK (1ULL << 14)
#define ZVL2048B_GROUPID 1
#define ZVL2048B_BITMASK (1ULL << 11)
#define ZVL256B_GROUPID 1
#define ZVL256B_BITMASK (1ULL << 8)
#define ZVL32768B_GROUPID 1
#define ZVL32768B_BITMASK (1ULL << 15)
#define ZVL32B_GROUPID 1
#define ZVL32B_BITMASK (1ULL << 5)
#define ZVL4096B_GROUPID 1
#define ZVL4096B_BITMASK (1ULL << 12)
#define ZVL512B_GROUPID 1
#define ZVL512B_BITMASK (1ULL << 9)
#define ZVL64B_GROUPID 1
#define ZVL64B_BITMASK (1ULL << 6)
#define ZVL65536B_GROUPID 1
#define ZVL65536B_BITMASK (1ULL << 16)
#define ZVL8192B_GROUPID 1
#define ZVL8192B_BITMASK (1ULL << 13)

#define HWPROBE_LENGTH 3

static void initRISCVFeature(struct riscv_hwprobe Hwprobes[]) {

  // Init vendor extension
  __riscv_vendor_feature_bits.length = 0;
  __riscv_vendor_feature_bits.vendorID = Hwprobes[2].value;

  // Init standard extension
  // TODO: Maybe Extension implied generate from tablegen?
  __riscv_feature_bits.length = 2;
  // Check RISCV_HWPROBE_KEY_BASE_BEHAVIOR
  unsigned long long BaseValue = Hwprobes[0].value;
  if (BaseValue & RISCV_HWPROBE_BASE_BEHAVIOR_IMA) {
    __riscv_feature_bits.features[I_GROUPID] |= I_BITMASK;
    __riscv_feature_bits.features[M_GROUPID] |= M_BITMASK;
    __riscv_feature_bits.features[A_GROUPID] |= A_BITMASK;
  }

  // Check RISCV_HWPROBE_KEY_IMA_EXT_0
  unsigned long long IMAEXT0Value = Hwprobes[1].value;
  if (IMAEXT0Value & RISCV_HWPROBE_IMA_FD) {
    __riscv_feature_bits.features[F_GROUPID] |= F_BITMASK;
    __riscv_feature_bits.features[D_GROUPID] |= D_BITMASK;
    __riscv_feature_bits.features[ZICSR_GROUPID] |= ZICSR_BITMASK;
  }

  if (IMAEXT0Value & RISCV_HWPROBE_IMA_C) {
    __riscv_feature_bits.features[C_GROUPID] |= C_BITMASK;
  }

  if (IMAEXT0Value & RISCV_HWPROBE_IMA_V) {
    __riscv_feature_bits.features[V_GROUPID] |= V_BITMASK;
    __riscv_feature_bits.features[ZVE32X_GROUPID] |= ZVE32X_BITMASK;
    __riscv_feature_bits.features[ZVE64X_GROUPID] |= ZVE64X_BITMASK;
    __riscv_feature_bits.features[ZVL32B_GROUPID] |= ZVL32B_BITMASK;
    __riscv_feature_bits.features[F_GROUPID] |= F_BITMASK;
    __riscv_feature_bits.features[ZVE64D_GROUPID] |= ZVE64D_BITMASK;
    __riscv_feature_bits.features[D_GROUPID] |= D_BITMASK;
    __riscv_feature_bits.features[ZVE32F_GROUPID] |= ZVE32F_BITMASK;
    __riscv_feature_bits.features[ZVL128B_GROUPID] |= ZVL128B_BITMASK;
    __riscv_feature_bits.features[ZVE64F_GROUPID] |= ZVE64F_BITMASK;
    __riscv_feature_bits.features[ZICSR_GROUPID] |= ZICSR_BITMASK;
    __riscv_feature_bits.features[ZVL64B_GROUPID] |= ZVL64B_BITMASK;
  }

  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZBA) {
    __riscv_feature_bits.features[ZBA_GROUPID] |= ZBA_BITMASK;
  }

  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZBB) {
    __riscv_feature_bits.features[ZBB_GROUPID] |= ZBB_BITMASK;
  }

  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZBS) {
    __riscv_feature_bits.features[ZBS_GROUPID] |= ZBS_BITMASK;
  }

  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZICBOZ) {
    __riscv_feature_bits.features[ZICBOZ_GROUPID] |= ZICBOZ_BITMASK;
  }

  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZBC) {
    __riscv_feature_bits.features[ZBC_GROUPID] |= ZBC_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZBKB) {
    __riscv_feature_bits.features[ZBKB_GROUPID] |= ZBKB_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZBKC) {
    __riscv_feature_bits.features[ZBKC_GROUPID] |= ZBKC_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZBKX) {
    __riscv_feature_bits.features[ZBKX_GROUPID] |= ZBKX_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZKND) {
    __riscv_feature_bits.features[ZKND_GROUPID] |= ZKND_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZKNE) {
    __riscv_feature_bits.features[ZKNE_GROUPID] |= ZKNE_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZKNH) {
    __riscv_feature_bits.features[ZKNH_GROUPID] |= ZKNH_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZKSED) {
    __riscv_feature_bits.features[ZKSED_GROUPID] |= ZKSED_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZKSH) {
    __riscv_feature_bits.features[ZKSH_GROUPID] |= ZKSH_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZKT) {
    __riscv_feature_bits.features[ZKT_GROUPID] |= ZKT_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZVBB) {
    __riscv_feature_bits.features[ZVBB_GROUPID] |= ZVBB_BITMASK;
    __riscv_feature_bits.features[ZVKB_GROUPID] |= ZVKB_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZVBC) {
    __riscv_feature_bits.features[ZVBC_GROUPID] |= ZVBC_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZVKB) {
    __riscv_feature_bits.features[ZVKB_GROUPID] |= ZVKB_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZVKG) {
    __riscv_feature_bits.features[ZVKG_GROUPID] |= ZVKG_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZVKNED) {
    __riscv_feature_bits.features[ZVKNED_GROUPID] |= ZVKNED_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZVKNHA) {
    __riscv_feature_bits.features[ZVKNHA_GROUPID] |= ZVKNHA_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZVKNHB) {
    __riscv_feature_bits.features[ZVKNHB_GROUPID] |= ZVKNHB_BITMASK;
    __riscv_feature_bits.features[ZVE32X_GROUPID] |= ZVE32X_BITMASK;
    __riscv_feature_bits.features[ZVE64X_GROUPID] |= ZVE64X_BITMASK;
    __riscv_feature_bits.features[ZVL32B_GROUPID] |= ZVL32B_BITMASK;
    __riscv_feature_bits.features[ZICSR_GROUPID] |= ZICSR_BITMASK;
    __riscv_feature_bits.features[ZVL64B_GROUPID] |= ZVL64B_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZVKSED) {
    __riscv_feature_bits.features[ZVKSED_GROUPID] |= ZVKSED_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZVKSH) {
    __riscv_feature_bits.features[ZVKSH_GROUPID] |= ZVKSH_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZVKT) {
    __riscv_feature_bits.features[ZVKT_GROUPID] |= ZVKT_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZFH) {
    __riscv_feature_bits.features[ZFH_GROUPID] |= ZFH_BITMASK;
    __riscv_feature_bits.features[F_GROUPID] |= F_BITMASK;
    __riscv_feature_bits.features[ZFHMIN_GROUPID] |= ZFHMIN_BITMASK;
    __riscv_feature_bits.features[ZICSR_GROUPID] |= ZICSR_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZFHMIN) {
    __riscv_feature_bits.features[ZFHMIN_GROUPID] |= ZFHMIN_BITMASK;
    __riscv_feature_bits.features[F_GROUPID] |= F_BITMASK;
    __riscv_feature_bits.features[ZICSR_GROUPID] |= ZICSR_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZIHINTNTL) {
    __riscv_feature_bits.features[ZIHINTNTL_GROUPID] |= ZIHINTNTL_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZVFH) {
    __riscv_feature_bits.features[ZVFH_GROUPID] |= ZVFH_BITMASK;
    __riscv_feature_bits.features[ZVE32X_GROUPID] |= ZVE32X_BITMASK;
    __riscv_feature_bits.features[ZVL32B_GROUPID] |= ZVL32B_BITMASK;
    __riscv_feature_bits.features[F_GROUPID] |= F_BITMASK;
    __riscv_feature_bits.features[ZVFHMIN_GROUPID] |= ZVFHMIN_BITMASK;
    __riscv_feature_bits.features[ZVE32F_GROUPID] |= ZVE32F_BITMASK;
    __riscv_feature_bits.features[ZFHMIN_GROUPID] |= ZFHMIN_BITMASK;
    __riscv_feature_bits.features[ZICSR_GROUPID] |= ZICSR_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZVFHMIN) {
    __riscv_feature_bits.features[ZVFHMIN_GROUPID] |= ZVFHMIN_BITMASK;
    __riscv_feature_bits.features[ZVE32X_GROUPID] |= ZVE32X_BITMASK;
    __riscv_feature_bits.features[ZVL32B_GROUPID] |= ZVL32B_BITMASK;
    __riscv_feature_bits.features[F_GROUPID] |= F_BITMASK;
    __riscv_feature_bits.features[ZVE32F_GROUPID] |= ZVE32F_BITMASK;
    __riscv_feature_bits.features[ZICSR_GROUPID] |= ZICSR_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZFA) {
    __riscv_feature_bits.features[ZFA_GROUPID] |= ZFA_BITMASK;
    __riscv_feature_bits.features[F_GROUPID] |= F_BITMASK;
    __riscv_feature_bits.features[ZICSR_GROUPID] |= ZICSR_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZTSO) {
    __riscv_feature_bits.features[ZTSO_GROUPID] |= ZTSO_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZACAS) {
    __riscv_feature_bits.features[ZACAS_GROUPID] |= ZACAS_BITMASK;
  }
  if (IMAEXT0Value & RISCV_HWPROBE_EXT_ZICOND) {
    __riscv_feature_bits.features[ZICOND_GROUPID] |= ZICOND_BITMASK;
  }
}

static int FeaturesBitCached = 0;

void __init_riscv_features_bit() {

  if (FeaturesBitCached)
    return;

  FeaturesBitCached = 1;

  struct riscv_hwprobe Hwprobes[HWPROBE_LENGTH];
  Hwprobes[0].key = RISCV_HWPROBE_KEY_BASE_BEHAVIOR;
  Hwprobes[1].key = RISCV_HWPROBE_KEY_IMA_EXT_0;
  Hwprobes[2].key = RISCV_HWPROBE_KEY_MVENDORID;
  initHwProbe(Hwprobes, HWPROBE_LENGTH);

  initRISCVFeature(Hwprobes);
}
