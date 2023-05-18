// REQUIRES: powerpc-registered-target
// RUN: not %clang_cc1 -triple powerpc64-unknown-unknown \
// RUN: -target-cpu pwr6 -emit-llvm %s -o - 2>&1 \
// RUN: | FileCheck %s

// RUN: not %clang_cc1 -triple powerpc-unknown-unknown -emit-llvm %s -o - 2>&1 \
// RUN: -target-cpu pwr7 | FileCheck %s -check-prefix=CHECK-32

// CHECK: error: use of '__int128' with '__vector' requires extended Altivec support (available on POWER8 or later)
vector signed __int128 vslll = {33};

void call_p7_builtins(void)
{
  int a = __builtin_divwe(33, 11);
  unsigned int b = __builtin_divweu(33U, 11U);
  unsigned long long d = __builtin_divde(33ULL, 11ULL);
  unsigned long long e = __builtin_divdeu(33ULL, 11ULL);
  unsigned long long f = __builtin_bpermd(33ULL, 11ULL);
  __builtin_pack_vector_int128(33ULL, 11ULL);
  __builtin_unpack_vector_int128(vslll, 1);
}

// CHECK-32: error: this builtin is only available on 64-bit targets
// CHECK-32: __builtin_divde
// CHECK-32: error: this builtin is only available on 64-bit targets
// CHECK-32: __builtin_divdeu
// CHECK-32: error: this builtin is only available on 64-bit targets
// CHECK-32: __builtin_bpermd
