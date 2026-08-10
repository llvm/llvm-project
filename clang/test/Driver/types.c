// Check whether __int128_t and __uint128_t are supported.

// RUN: not %clang -c --target=riscv32-unknown-linux-gnu -fsyntax-only %s \
// RUN: 2>&1 | FileCheck %s

// RUN: %clang -c --target=riscv32-unknown-linux-gnu -fsyntax-only %s \
// RUN: -fno-force-enable-int128 -fforce-enable-int128

// RUN: not %clang -c --target=riscv32-unknown-linux-gnu -fsyntax-only %s \
// RUN: -fforce-enable-int128 -fno-force-enable-int128

// RUN: not %clang -c --target=powerpc-ibm-aix -fsyntax-only %s \
// RUN: 2>&1 | FileCheck %s

// RUN: %clang -c --target=powerpc64-ibm-aix -fsyntax-only %s \
// RUN: 2>&1

void a(void) {
  __int128_t s;
  __uint128_t t;
}

// CHECK: error: use of undeclared identifier '__int128_t'
// CHECK: error: use of undeclared identifier '__uint128_t'
