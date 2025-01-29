// RUN: %clang_cc1 -triple armv4 %s -E -dD -o - | FileCheck --check-prefix=CHECK-V4 %s
// RUN: %clang_cc1 -triple armv4t %s -E -dD -o - | FileCheck --check-prefix=CHECK-V4 %s
// RUN: %clang_cc1 -triple armv5 %s -E -dD -o - | FileCheck --check-prefix=CHECK-V5 %s
// RUN: %clang_cc1 -triple armv5te %s -E -dD -o - | FileCheck --check-prefix=CHECK-V5-TE %s
// RUN: %clang_cc1 -triple armv5tej %s -E -dD -o - | FileCheck --check-prefix=CHECK-V5-TE %s
// RUN: %clang_cc1 -triple armv6 %s -E -dD -o - | FileCheck --check-prefix=CHECK-V6 %s
// RUN: %clang_cc1 -triple armv6m %s -E -dD -o - | FileCheck --check-prefix=CHECK-V6M %s
// RUN: %clang_cc1 -triple armv7a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V7 %s
// RUN: %clang_cc1 -triple armv7r %s -E -dD -o - | FileCheck --check-prefix=CHECK-V7 %s
// RUN: %clang_cc1 -triple armv7m %s -E -dD -o - | FileCheck --check-prefix=CHECK-V7 %s
// RUN: %clang_cc1 -triple armv8a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv8r %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv8.1a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv8.2a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv8.3a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv8.4a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv8.5a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv8.6a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv8.7a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv8.8a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv8.9a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv9a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv9.1a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv9.2a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv9.3a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv9.4a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv9.5a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple armv9.6a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv4 %s -E -dD -o - | FileCheck --check-prefix=CHECK-V4-THUMB %s
// RUN: %clang_cc1 -triple thumbv4t %s -E -dD -o - | FileCheck --check-prefix=CHECK-V4-THUMB %s
// RUN: %clang_cc1 -triple thumbv5 %s -E -dD -o - | FileCheck --check-prefix=CHECK-V5-THUMB %s
// RUN: %clang_cc1 -triple thumbv5te %s -E -dD -o - | FileCheck --check-prefix=CHECK-V5-TE-THUMB %s
// RUN: %clang_cc1 -triple thumbv5tej %s -E -dD -o - | FileCheck --check-prefix=CHECK-V5-TE-THUMB %s
// RUN: %clang_cc1 -triple thumbv6 %s -E -dD -o - | FileCheck --check-prefix=CHECK-V6-THUMB %s
// RUN: %clang_cc1 -triple thumbv6k %s -E -dD -o - | FileCheck --check-prefix=CHECK-V6-THUMB %s
// RUN: %clang_cc1 -triple thumbv6kz %s -E -dD -o - | FileCheck --check-prefix=CHECK-V6-THUMB %s
// RUN: %clang_cc1 -triple thumbv6m %s -E -dD -o - | FileCheck --check-prefix=CHECK-V6M %s
// RUN: %clang_cc1 -triple thumbv7a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V7 %s
// RUN: %clang_cc1 -triple thumbv7r %s -E -dD -o - | FileCheck --check-prefix=CHECK-V7 %s
// RUN: %clang_cc1 -triple thumbv7m %s -E -dD -o - | FileCheck --check-prefix=CHECK-V7 %s
// RUN: %clang_cc1 -triple thumbv8a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv8r %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv8.1a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv8.2a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv8.3a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv8.4a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv8.5a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv8.6a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv8.7a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv8.8a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv8.9a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv9a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv9.1a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv9.2a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv9.3a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv9.4a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv9.5a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv9.6a %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8 %s
// RUN: %clang_cc1 -triple thumbv8m.base %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8-BASE %s
// RUN: %clang_cc1 -triple thumbv8m.main %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8-MAIN %s
// RUN: %clang_cc1 -triple thumbv8.1m.main %s -E -dD -o - | FileCheck --check-prefix=CHECK-V8-MAIN %s

#include <arm_acle.h>

void cdp() {
  __arm_cdp(1, 2, 3, 4, 5, 6);
  // CHECK-LABEL: void cdp()
  // CHECK-V4: __builtin_arm_cdp
  // CHECK-V4-THUMB-NOT: __builtin_arm_cdp
  // CHECK-V5: __builtin_arm_cdp
  // CHECK-V5-TE: __builtin_arm_cdp
  // CHECK-V5-THUMB-NOT: __builtin_arm_cdp
  // CHECK-V5-TE-THUMB-NOT: __builtin_arm_cdp
  // CHECK-V6: __builtin_arm_cdp
  // CHECK-V6-THUMB-NOT: __builtin_arm_cdp
  // CHECK-V6M-NOT: __builtin_arm_cdp
  // CHECK-V7: __builtin_arm_cdp
  // CHECK-V8-NOT: __builtin_arm_cdp
  // CHECK-V8-BASE-NOT: __builtin_arm_cdp
  // CHECK-V8-MAIN: __builtin_arm_cdp
}

void cdp2() {
  __arm_cdp2(1, 2, 3, 4, 5, 6);
  // CHECK-LABEL: void cdp2()
  // CHECK-V4-NOT: __builtin_arm_cdp2
  // CHECK-V4-THUMB-NOT: __builtin_arm_cdp2
  // CHECK-V5: __builtin_arm_cdp2
  // CHECK-V5-TE: __builtin_arm_cdp2
  // CHECK-V5-THUMB-NOT: __builtin_arm_cdp2
  // CHECK-V5-TE-THUMB-NOT: __builtin_arm_cdp2
  // CHECK-V6: __builtin_arm_cdp2
  // CHECK-V6-THUMB-NOT: __builtin_arm_cdp2
  // CHECK-V6M-NOT: __builtin_arm_cdp2
  // CHECK-V7: __builtin_arm_cdp2
  // CHECK-V8-NOT: __builtin_arm_cdp2
  // CHECK-V8-BASE-NOT: __builtin_arm_cdp2
  // CHECK-V8-MAIN: __builtin_arm_cdp2
}

void ldc(int i) {
  __arm_ldc(1, 2, &i);
  // CHECK-LABEL: void ldc()
  // CHECK-V4: __builtin_arm_ldc
  // CHECK-V4-THUMB-NOT: __builtin_arm_ldc
  // CHECK-V5: __builtin_arm_ldc
  // CHECK-V5-TE: __builtin_arm_ldc
  // CHECK-V5-THUMB-NOT: __builtin_arm_ldc
  // CHECK-V5-TE-THUMB-NOT: __builtin_arm_ldc
  // CHECK-V6: __builtin_arm_ldc
  // CHECK-V6-THUMB-NOT: __builtin_arm_ldc
  // CHECK-V6M-NOT: __builtin_arm_ldc
  // CHECK-V7: __builtin_arm_ldc
  // CHECK-V8: __builtin_arm_ldc
  // CHECK-V8-BASE-NOT: __builtin_arm_ldc
  // CHECK-V8-MAIN: __builtin_arm_ldc
}

void ldcl(int i) {
  __arm_ldcl(1, 2, &i);
  // CHECK-LABEL: void ldcl()
  // CHECK-V4-NOT: __builtin_arm_ldcl
  // CHECK-V4-THUMB-NOT: __builtin_arm_ldcl
  // CHECK-V5: __builtin_arm_ldcl
  // CHECK-V5-TE: __builtin_arm_ldcl
  // CHECK-V5-THUMB-NOT: __builtin_arm_ldcl
  // CHECK-V5-TE-THUMB-NOT: __builtin_arm_ldcl
  // CHECK-V6: __builtin_arm_ldcl
  // CHECK-V6-THUMB-NOT: __builtin_arm_ldcl
  // CHECK-V6M-NOT: __builtin_arm_ldcl
  // CHECK-V7: __builtin_arm_ldcl
  // CHECK-V8-NOT: __builtin_arm_ldcl
  // CHECK-V8-BASE-NOT: __builtin_arm_ldcl
  // CHECK-V8-MAIN: __builtin_arm_ldcl
}

void ldc2(int i) {
  __arm_ldc2(1, 2, &i);
  // CHECK-LABEL: void ldc2()
  // CHECK-V4-NOT: __builtin_arm_ldc2
  // CHECK-V4-THUMB-NOT: __builtin_arm_ldc2
  // CHECK-V5: __builtin_arm_ldc2
  // CHECK-V5-TE: __builtin_arm_ldc2
  // CHECK-V5-THUMB-NOT: __builtin_arm_ldc2
  // CHECK-V5-TE-THUMB-NOT: __builtin_arm_ldc2
  // CHECK-V6: __builtin_arm_ldc2
  // CHECK-V6-THUMB-NOT: __builtin_arm_ldc2
  // CHECK-V6M-NOT: __builtin_arm_ldc2
  // CHECK-V7: __builtin_arm_ldc2
  // CHECK-V8-NOT: __builtin_arm_ldc2
  // CHECK-V8-BASE-NOT: __builtin_arm_ldc2
  // CHECK-V8-MAIN: __builtin_arm_ldc2
}

void ldc2l(int i) {
  __arm_ldc2l(1, 2, &i);
  // CHECK-LABEL: void ldc2l()
  // CHECK-V4-NOT: __builtin_arm_ldc2l
  // CHECK-V4-THUMB-NOT: __builtin_arm_ldc2l
  // CHECK-V5: __builtin_arm_ldc2l
  // CHECK-V5-TE: __builtin_arm_ldc2l
  // CHECK-V5-THUMB-NOT: __builtin_arm_ldc2l
  // CHECK-V5-TE-THUMB-NOT: __builtin_arm_ldc2l
  // CHECK-V6: __builtin_arm_ldc2l
  // CHECK-V6-THUMB-NOT: __builtin_arm_ldc2l
  // CHECK-V6M-NOT: __builtin_arm_ldc2l
  // CHECK-V7: __builtin_arm_ldc2l
  // CHECK-V8-NOT: __builtin_arm_ldc2l
  // CHECK-V8-BASE-NOT: __builtin_arm_ldc2l
  // CHECK-V8-MAIN: __builtin_arm_ldc2l
}

void stc(int i) {
  __arm_stc(1, 2, &i);
  // CHECK-LABEL: void stc()
  // CHECK-V4: __builtin_arm_stc
  // CHECK-V4-THUMB-NOT: __builtin_arm_stc
  // CHECK-V5: __builtin_arm_stc
  // CHECK-V5-TE: __builtin_arm_stc
  // CHECK-V5-THUMB-NOT: __builtin_arm_stc
  // CHECK-V5-TE-THUMB-NOT: __builtin_arm_stc
  // CHECK-V6: __builtin_arm_stc
  // CHECK-V6-THUMB-NOT: __builtin_arm_stc
  // CHECK-V6M-NOT: __builtin_arm_stc
  // CHECK-V7: __builtin_arm_stc
  // CHECK-V8: __builtin_arm_stc
  // CHECK-V8-BASE-NOT: __builtin_arm_stc
  // CHECK-V8-MAIN: __builtin_arm_stc
}

void stcl(int i) {
  __arm_stcl(1, 2, &i);
  // CHECK-LABEL: void stcl()
  // CHECK-V4-NOT: __builtin_arm_stcl
  // CHECK-V4-THUMB-NOT: __builtin_arm_stcl
  // CHECK-V5: __builtin_arm_stcl
  // CHECK-V5-TE: __builtin_arm_stcl
  // CHECK-V5-THUMB-NOT: __builtin_arm_stcl
  // CHECK-V5-TE-THUMB-NOT: __builtin_arm_stcl
  // CHECK-V6: __builtin_arm_stcl
  // CHECK-V6-THUMB-NOT: __builtin_arm_stcl
  // CHECK-V6M-NOT: __builtin_arm_stcl
  // CHECK-V7: __builtin_arm_stcl
  // CHECK-V8-NOT: __builtin_arm_stcl
  // CHECK-V8-BASE-NOT: __builtin_arm_stcl
  // CHECK-V8-MAIN: __builtin_arm_stcl
}

void stc2(int i) {
  __arm_stc2(1, 2, &i);
  // CHECK-LABEL: void stc2()
  // CHECK-V4-NOT: __builtin_arm_stc2
  // CHECK-V4-THUMB-NOT: __builtin_arm_stc2
  // CHECK-V5: __builtin_arm_stc2
  // CHECK-V5-TE: __builtin_arm_stc2
  // CHECK-V5-THUMB-NOT: __builtin_arm_stc2
  // CHECK-V5-TE-THUMB-NOT: __builtin_arm_stc2
  // CHECK-V6: __builtin_arm_stc2
  // CHECK-V6-THUMB-NOT: __builtin_arm_stc2
  // CHECK-V6M-NOT: __builtin_arm_stc2
  // CHECK-V7: __builtin_arm_stc2
  // CHECK-V8-NOT: __builtin_arm_stc2
  // CHECK-V8-BASE-NOT: __builtin_arm_stc2
  // CHECK-V8-MAIN: __builtin_arm_stc2
}

void stc2l(int i) {
  __arm_stc2l(1, 2, &i);
  // CHECK-LABEL: void stc2l()
  // CHECK-V4-NOT: __builtin_arm_stc2l
  // CHECK-V4-THUMB-NOT: __builtin_arm_stc2l
  // CHECK-V5: __builtin_arm_stc2l
  // CHECK-V5-TE: __builtin_arm_stc2l
  // CHECK-V5-THUMB-NOT: __builtin_arm_stc2l
  // CHECK-V5-TE-THUMB-NOT: __builtin_arm_stc2l
  // CHECK-V6: __builtin_arm_stc2l
  // CHECK-V6-THUMB-NOT: __builtin_arm_stc2l
  // CHECK-V6M-NOT: __builtin_arm_stc2l
  // CHECK-V7: __builtin_arm_stc2l
  // CHECK-V8-NOT: __builtin_arm_stc2l
  // CHECK-V8-BASE-NOT: __builtin_arm_stc2l
  // CHECK-V8-MAIN: __builtin_arm_stc2l
}

void mcr() {
  __arm_mcr(1, 2, 3, 4, 5, 6);
  // CHECK-LABEL: void mcr()
  // CHECK-V4: __builtin_arm_mcr
  // CHECK-V4-THUMB-NOT: __builtin_arm_mcr
  // CHECK-V5: __builtin_arm_mcr
  // CHECK-V5-TE: __builtin_arm_mcr
  // CHECK-V5-THUMB-NOT: __builtin_arm_mcr
  // CHECK-V5-TE-THUMB-NOT: __builtin_arm_mcr
  // CHECK-V6: __builtin_arm_mcr
  // CHECK-V6-THUMB-NOT: __builtin_arm_mcr
  // CHECK-V6M-NOT: __builtin_arm_mcr
  // CHECK-V7: __builtin_arm_mcr
  // CHECK-V8: __builtin_arm_mcr
  // CHECK-V8-BASE-NOT: __builtin_arm_mcr
  // CHECK-V8-MAIN: __builtin_arm_mcr
}

void mcr2() {
  __arm_mcr2(1, 2, 3, 4, 5, 6);
  // CHECK-LABEL: void mcr2()
  // CHECK-V4-NOT: __builtin_arm_mcr2
  // CHECK-V4-THUMB-NOT: __builtin_arm_mcr2
  // CHECK-V5: __builtin_arm_mcr2
  // CHECK-V5-TE: __builtin_arm_mcr2
  // CHECK-V5-THUMB-NOT: __builtin_arm_mcr2
  // CHECK-V5-TE-THUMB-NOT: __builtin_arm_mcr2
  // CHECK-V6: __builtin_arm_mcr2
  // CHECK-V6-THUMB-NOT: __builtin_arm_mcr2
  // CHECK-V6M-NOT: __builtin_arm_mcr2
  // CHECK-V7: __builtin_arm_mcr2
  // CHECK-V8-NOT: __builtin_arm_mcr2
  // CHECK-V8-BASE-NOT: __builtin_arm_mcr2
  // CHECK-V8-MAIN: __builtin_arm_mcr2
}

void mrc() {
  __arm_mrc(1, 2, 3, 4, 5);
  // CHECK-LABEL: void mrc()
  // CHECK-V4: __builtin_arm_mrc
  // CHECK-V4-THUMB-NOT: __builtin_arm_mrc
  // CHECK-V5: __builtin_arm_mrc
  // CHECK-V5-TE: __builtin_arm_mrc
  // CHECK-V5-THUMB-NOT: __builtin_arm_mrc
  // CHECK-V5-TE-THUMB-NOT: __builtin_arm_mrc
  // CHECK-V6: __builtin_arm_mrc
  // CHECK-V6-THUMB-NOT: __builtin_arm_mrc
  // CHECK-V6M-NOT: __builtin_arm_mrc
  // CHECK-V7: __builtin_arm_mrc
  // CHECK-V8: __builtin_arm_mrc
  // CHECK-V8-BASE-NOT: __builtin_arm_mrc
  // CHECK-V8-MAIN: __builtin_arm_mrc
}

void mrc2() {
  __arm_mrc2(1, 2, 3, 4, 5);
  // CHECK-LABEL: void mrc2()
  // CHECK-V4-NOT: __builtin_arm_mrc2
  // CHECK-V4-THUMB-NOT: __builtin_arm_mrc2
  // CHECK-V5: __builtin_arm_mrc2
  // CHECK-V5-TE: __builtin_arm_mrc2
  // CHECK-V5-THUMB-NOT: __builtin_arm_mrc2
  // CHECK-V5-TE-THUMB-NOT: __builtin_arm_mrc2
  // CHECK-V6: __builtin_arm_mrc2
  // CHECK-V6-THUMB-NOT: __builtin_arm_mrc2
  // CHECK-V6M-NOT: __builtin_arm_mrc2
  // CHECK-V7: __builtin_arm_mrc2
  // CHECK-V8-NOT: __builtin_arm_mrc2
  // CHECK-V8-BASE-NOT: __builtin_arm_mrc2
  // CHECK-V8-MAIN: __builtin_arm_mrc2
}

void mcrr() {
  __arm_mcrr(1, 2, 3, 4);
  // CHECK-LABEL: void mcrr()
  // CHECK-V4-NOT: __builtin_arm_mcrr
  // CHECK-V4-THUMB-NOT: __builtin_arm_mcrr
  // CHECK-V5-NOT: __builtin_arm_mcrr
  // CHECK-V5-TE: __builtin_arm_mcrr
  // CHECK-V5-THUMB-NOT: __builtin_arm_mcrr
  // CHECK-V5-THUMB-NOT: __builtin_arm_mcrr
  // CHECK-V6: __builtin_arm_mcrr
  // CHECK-V6-THUMB-NOT: __builtin_arm_mcrr
  // CHECK-V6M-NOT: __builtin_arm_mcrr
  // CHECK-V7: __builtin_arm_mcrr
  // CHECK-V8: __builtin_arm_mcrr
  // CHECK-V8-BASE-NOT: __builtin_arm_mcrr
  // CHECK-V8-MAIN: __builtin_arm_mcrr
}

void mcrr2() {
  __arm_mcrr2(1, 2, 3, 4);
  // CHECK-LABEL: void mcrr2()
  // CHECK-V4-NOT: __builtin_arm_mcrr2
  // CHECK-V4-THUMB-NOT: __builtin_arm_mcrr2
  // CHECK-V5-NOT: __builtin_arm_mcrr2
  // CHECK-V5-TE-NOT: __builtin_arm_mcrr2
  // CHECK-V5-THUMB-NOT: __builtin_arm_mcrr2
  // CHECK-V5-TE-THUMB-NOT: __builtin_arm_mcrr2
  // CHECK-V6: __builtin_arm_mcrr2
  // CHECK-V6-THUMB-NOT: __builtin_arm_mcrr2
  // CHECK-V6M-NOT: __builtin_arm_mcrr2
  // CHECK-V7: __builtin_arm_mcrr2
  // CHECK-V8-NOT: __builtin_arm_mcrr2
  // CHECK-V8-BASE-NOT: __builtin_arm_mcrr2
  // CHECK-V8-MAIN: __builtin_arm_mcrr2
}

void mrrc() {
  __arm_mrrc(1, 2, 3);
  // CHECK-LABEL: void mrrc()
  // CHECK-V4-NOT: __builtin_arm_mrrc
  // CHECK-V4-THUMB-NOT: __builtin_arm_mrrc
  // CHECK-V5-NOT: __builtin_arm_mrrc
  // CHECK-V5-TE: __builtin_arm_mrrc
  // CHECK-V5-THUMB-NOT: __builtin_arm_mrrc
  // CHECK-V5-THUMB-TE-NOT: __builtin_arm_mrrc
  // CHECK-V6: __builtin_arm_mrrc
  // CHECK-V6-THUMB-NOT: __builtin_arm_mrrc
  // CHECK-V6M-NOT: __builtin_arm_mrrc
  // CHECK-V7: __builtin_arm_mrrc
  // CHECK-V8: __builtin_arm_mrrc
  // CHECK-V8-BASE-NOT: __builtin_arm_mrrc
  // CHECK-V8-MAIN: __builtin_arm_mrrc
}

void mrrc2() {
  __arm_mrrc2(1, 2, 3);
  // CHECK-LABEL: void mrrc2()
  // CHECK-V4-NOT: __builtin_arm_mrrc2
  // CHECK-V4-THUMB-NOT: __builtin_arm_mrrc2
  // CHECK-V5-NOT: __builtin_arm_mrrc2
  // CHECK-V5-TE-NOT: __builtin_arm_mrrc2
  // CHECK-V5-THUMB-NOT: __builtin_arm_mrrc2
  // CHECK-V5-TE-THUMB-NOT: __builtin_arm_mrrc2
  // CHECK-V6: __builtin_arm_mrrc2
  // CHECK-V6-THUMB-NOT: __builtin_arm_mrrc2
  // CHECK-V6M-NOT: __builtin_arm_mrrc2
  // CHECK-V7: __builtin_arm_mrrc2
  // CHECK-V8-NOT: __builtin_arm_mrrc2
  // CHECK-V8-BASE-NOT: __builtin_arm_mrrc2
  // CHECK-V8-MAIN: __builtin_arm_mrrc2
}
