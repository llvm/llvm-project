// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -O2 -target-feature +outline-atomics -target-feature +outline-atomics-preserve-most -emit-llvm -o - %s | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -O2 -target-feature +outline-atomics -target-feature +outline-atomics-preserve-most -S -o - %s | FileCheck %s --check-prefix=ASM
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -O2 -target-feature +outline-atomics -mllvm --aarch64-outline-atomics-preserve-most -S -o - %s | FileCheck %s --check-prefix=ASM
// RUN: %clang_cc1 -triple aarch64-unknown-fuchsia -O2 -target-feature +outline-atomics -target-feature +outline-atomics-preserve-most -emit-llvm -o - %s | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -triple aarch64-unknown-fuchsia -O2 -target-feature +outline-atomics -target-feature +outline-atomics-preserve-most -S -o - %s | FileCheck %s --check-prefix=ASM
// RUN: %clang_cc1 -triple aarch64-unknown-fuchsia -O2 -target-feature +outline-atomics -mllvm --aarch64-outline-atomics-preserve-most -S -o - %s | FileCheck %s --check-prefix=ASM

int fetch_add_with_live_vals(_Atomic int *p, int v, int a, int b, int c, int d,
                             int e, int f) {
  // ASM-LABEL: fetch_add_with_live_vals:
  // ASM-NOT: stp x19
  // ASM-NOT: stp x20
  // ASM: bl __aarch64_ldadd4_acq_rel_preservemost
  // ASM: ret
  __c11_atomic_fetch_add(p, v, __ATOMIC_ACQ_REL);
  return a + b + c + d + e + f;
}

// IR: attributes #{{[0-9]+}} = { {{.*}}"target-features"="{{.*}}+outline-atomics,+outline-atomics-preserve-most{{.*}}"
