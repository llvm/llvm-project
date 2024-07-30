//=== cpu_model/riscv.c - Update RISC-V Feature Bits Structure -*- C -*-======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cpu_model.h"

#define RISCV_FEATURE_BITS_LENGTH 1
struct {
  unsigned length;
  unsigned long long features[RISCV_FEATURE_BITS_LENGTH];
} __riscv_feature_bits __attribute__((visibility("hidden"), nocommon));

#define RISCV_VENDOR_FEATURE_BITS_LENGTH 1
struct {
  unsigned vendorID;
  unsigned length;
  unsigned long long features[RISCV_VENDOR_FEATURE_BITS_LENGTH];
} __riscv_vendor_feature_bits __attribute__((visibility("hidden"), nocommon));

// NOTE: Should sync-up with RISCVFeatures.td
// TODO: Maybe generate a header from tablegen then include it.
#define A_GROUPID 0
#define A_BITMASK (1ULL << 0)
#define C_GROUPID 0
#define C_BITMASK (1ULL << 2)
#define D_GROUPID 0
#define D_BITMASK (1ULL << 3)
#define F_GROUPID 0
#define F_BITMASK (1ULL << 5)
#define I_GROUPID 0
#define I_BITMASK (1ULL << 8)
#define M_GROUPID 0
#define M_BITMASK (1ULL << 12)
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

#if defined(__linux__)

// The RISC-V hwprobe interface is documented here:
// <https://docs.kernel.org/arch/riscv/hwprobe.html>.

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
#define RISCV_HWPROBE_EXT_ZIHINTPAUSE (1ULL << 36)
#define RISCV_HWPROBE_KEY_CPUPERF_0 5
#define RISCV_HWPROBE_MISALIGNED_UNKNOWN (0 << 0)
#define RISCV_HWPROBE_MISALIGNED_EMULATED (1ULL << 0)
#define RISCV_HWPROBE_MISALIGNED_SLOW (2 << 0)
#define RISCV_HWPROBE_MISALIGNED_FAST (3 << 0)
#define RISCV_HWPROBE_MISALIGNED_UNSUPPORTED (4 << 0)
#define RISCV_HWPROBE_MISALIGNED_MASK (7 << 0)
#define RISCV_HWPROBE_KEY_ZICBOZ_BLOCK_SIZE 6
/* Increase RISCV_HWPROBE_MAX_KEY when adding items. */

struct riscv_hwprobe {
  long long key;
  unsigned long long value;
};

#define __NR_riscv_hwprobe 258
static long initHwProbe(struct riscv_hwprobe *Hwprobes, int len) {
  return syscall_impl_5_args(__NR_riscv_hwprobe, (long)Hwprobes, len, 0, 0, 0);
}

#define SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(EXTNAME)                    \
  SET_SINGLE_IMAEXT_RISCV_FEATURE(RISCV_HWPROBE_EXT_##EXTNAME, EXTNAME)

#define SET_SINGLE_IMAEXT_RISCV_FEATURE(HWPROBE_BITMASK, EXT)                  \
  SET_SINGLE_RISCV_FEATURE(IMAEXT0Value &HWPROBE_BITMASK, EXT)

#define SET_SINGLE_RISCV_FEATURE(COND, EXT)                                    \
  if (COND) {                                                                  \
    SET_RISCV_FEATURE(EXT);                                                    \
  }

#define SET_RISCV_FEATURE(EXT) features[EXT##_GROUPID] |= EXT##_BITMASK

static void initRISCVFeature(struct riscv_hwprobe Hwprobes[]) {

  // Note: If a hwprobe key is unknown to the kernel, its key field
  // will be cleared to -1, and its value set to 0.
  // This unsets all extension bitmask bits.

  // Init vendor extension
  __riscv_vendor_feature_bits.vendorID = Hwprobes[2].value;

  // Init standard extension
  // TODO: Maybe Extension implied generate from tablegen?

  unsigned long long features[RISCV_FEATURE_BITS_LENGTH];
  int i;

  for (i = 0; i < RISCV_FEATURE_BITS_LENGTH; i++)
    features[i] = 0;

  // Check RISCV_HWPROBE_KEY_BASE_BEHAVIOR
  unsigned long long BaseValue = Hwprobes[0].value;
  if (BaseValue & RISCV_HWPROBE_BASE_BEHAVIOR_IMA) {
    SET_RISCV_FEATURE(I);
    SET_RISCV_FEATURE(M);
    SET_RISCV_FEATURE(A);
  }

  // Check RISCV_HWPROBE_KEY_IMA_EXT_0
  unsigned long long IMAEXT0Value = Hwprobes[1].value;
  if (IMAEXT0Value & RISCV_HWPROBE_IMA_FD) {
    SET_RISCV_FEATURE(F);
    SET_RISCV_FEATURE(D);
  }

  SET_SINGLE_IMAEXT_RISCV_FEATURE(RISCV_HWPROBE_IMA_C, C);
  SET_SINGLE_IMAEXT_RISCV_FEATURE(RISCV_HWPROBE_IMA_V, V);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZBA);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZBB);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZBS);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZICBOZ);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZBC);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZBKB);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZBKC);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZBKX);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZKND);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZKNE);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZKNH);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZKSED);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZKSH);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZKT);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZVBB);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZVBC);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZVKB);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZVKG);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZVKNED);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZVKNHA);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZVKNHB);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZVKSED);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZVKSH);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZVKT);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZFH);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZFHMIN);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZIHINTNTL);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZIHINTPAUSE);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZVFH);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZVFHMIN);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZFA);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZTSO);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZACAS);
  SET_RISCV_HWPROBE_EXT_SINGLE_RISCV_FEATURE(ZICOND);

  for (i = 0; i < RISCV_FEATURE_BITS_LENGTH; i++)
    __riscv_feature_bits.features[i] = features[i];
}

#endif // defined(__linux__)

static int FeaturesBitCached = 0;

void __init_riscv_feature_bits() CONSTRUCTOR_ATTRIBUTE;

// A constructor function that sets __riscv_feature_bits, and
// __riscv_vendor_feature_bits to the right values.  This needs to run
// only once.  This constructor is given the highest priority and it should
// run before constructors without the priority set.  However, it still runs
// after ifunc initializers and needs to be called explicitly there.
void CONSTRUCTOR_ATTRIBUTE __init_riscv_feature_bits() {

  if (FeaturesBitCached)
    return;

  __riscv_feature_bits.length = RISCV_FEATURE_BITS_LENGTH;
  __riscv_vendor_feature_bits.length = RISCV_VENDOR_FEATURE_BITS_LENGTH;

#if defined(__linux__)
  struct riscv_hwprobe Hwprobes[] = {
      {RISCV_HWPROBE_KEY_BASE_BEHAVIOR, 0},
      {RISCV_HWPROBE_KEY_IMA_EXT_0, 0},
      {RISCV_HWPROBE_KEY_MVENDORID, 0},
  };
  if (initHwProbe(Hwprobes, sizeof(Hwprobes) / sizeof(Hwprobes[0])))
    return;

  initRISCVFeature(Hwprobes);
#endif // defined(__linux__)

  FeaturesBitCached = 1;
}
