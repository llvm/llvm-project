// REQUIRES: powerpc-registered-target
// RUN: not %clang_cc1 -target-feature +altivec -triple powerpc64le-unknown-unknown \
// RUN: -target-cpu pwr8 -target-feature -crypto -emit-llvm %s -o - 2>&1 \
// RUN: | FileCheck %s

// RUN: not %clang_cc1 -target-feature +altivec -triple powerpc64-unknown-unknown \
// RUN: -target-cpu pwr8 -target-feature -crypto -emit-llvm %s -o - 2>&1 \
// RUN: | FileCheck %s

// RUN: not %clang_cc1 -target-feature +altivec -triple powerpc64-unknown-unknown \
// RUN: -target-cpu pwr8 -target-feature -power8-vector \
// RUN: -target-feature -crypto -emit-llvm %s -o - 2>&1 \
// RUN: | FileCheck %s -check-prefix=CHECK-P8V
#include <altivec.h>

#define W_INIT1 { 0x01020304, 0x05060708, \
                  0x090A0B0C, 0x0D0E0F10 };
#define D_INIT1 { 0x0102030405060708, \
                  0x090A0B0C0D0E0F10 };
#define D_INIT2 { 0x7172737475767778, \
                  0x797A7B7C7D7E7F70 };

// Test cases for the builtins the way they are exposed to
// users through altivec.h
void call_crypto_intrinsics(void)
{
  vector unsigned int aw = W_INIT1
  vector unsigned long long ad = D_INIT1
  vector unsigned long long bd = D_INIT2
  vector unsigned long long cd = D_INIT2

  vector unsigned long long r1 = __builtin_crypto_vsbox(ad);
  vector unsigned long long r2 = __builtin_crypto_vcipher(ad, bd);
  vector unsigned long long r3 = __builtin_crypto_vcipherlast(ad, bd);
  vector unsigned long long r4 = __builtin_crypto_vncipher(ad, bd);
  vector unsigned long long r5 = __builtin_crypto_vncipherlast(ad, bd);
  vector unsigned int       r6 = __builtin_crypto_vshasigmaw(aw, 1, 15);
  vector unsigned long long r7 = __builtin_crypto_vshasigmad(ad, 0, 15);

  // The ones that do not require -mcrypto, but require -mpower8-vector
  vector unsigned long long r8 = __builtin_crypto_vpmsumb(ad, bd);
  vector unsigned long long r9 = __builtin_crypto_vpermxor(ad, bd, cd);
}

// CHECK: use of unknown builtin '__builtin_crypto_vsbox'
// CHECK: use of unknown builtin '__builtin_crypto_vcipher'
// CHECK: use of unknown builtin '__builtin_crypto_vcipherlast'
// CHECK: use of unknown builtin '__builtin_crypto_vncipher'
// CHECK: use of unknown builtin '__builtin_crypto_vncipherlast'
// CHECK: use of unknown builtin '__builtin_crypto_vshasigmaw'
// CHECK: use of unknown builtin '__builtin_crypto_vshasigmad'
// CHECK-P8V: use of unknown builtin '__builtin_crypto_vpmsumb'
// CHECK-P8V: use of unknown builtin '__builtin_crypto_vpermxor'
