// REQUIRES: x86-registered-target
// REQUIRES: arm-registered-target
// REQUIRES: nvptx-registered-target

// Check -fsplit-machine-functions passed to cuda device causes a warning.
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_70 \
// RUN:     -fcuda-is-device -x cuda -fsplit-machine-functions -S %s \
// RUN:     -o %t 2>&1 | FileCheck %s --check-prefix=MFS1
// MFS1: warning: -fsplit-machine-functions is not valid for nvptx

// Check -fsplit-machine-functions passed to X86 does not cause any warning.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsplit-machine-functions \
// RUN:     -o %t -S %s && { echo "empty output causes FileCheck to fail" ; } \
// RUN:     2>&1 | FileCheck %s --check-prefix=MFS2
// MFS2-NOT: warning:

// Check -fsplit-machine-functions passed to ARM does cause a warning.
// RUN: %clang_cc1 -triple arm-unknown-linux-gnueabi \
// RUN:     -fsplit-machine-functions -S %s -o %t \
// RUN:     2>&1 | FileCheck -check-prefix=MFS3 %s
// MFS3: warning: -fsplit-machine-functions is not valid for arm

int foo() {
  return 13;
}
