// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +altivec -triple powerpc64le-unknown-unknown \
// RUN: -target-feature +crypto -target-feature +power8-vector \
// RUN: -emit-llvm %s -o - | FileCheck %s

// RUN: %clang_cc1 -target-feature +altivec -triple powerpc64-unknown-unknown \
// RUN: -target-feature +crypto -target-feature +power8-vector \
// RUN: -emit-llvm %s -o - | FileCheck %s
#include <altivec.h>
#define B_INIT1 { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, \
                  0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10 };
#define B_INIT2 { 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, \
                  0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F, 0x70 };
#define H_INIT1 { 0x0102, 0x0304, 0x0506, 0x0708, \
                  0x090A, 0x0B0C, 0x0D0E, 0x0F10 };
#define H_INIT2 { 0x7172, 0x7374, 0x7576, 0x7778, \
                  0x797A, 0x7B7C, 0x7D7E, 0x7F70 };
#define W_INIT1 { 0x01020304, 0x05060708, \
                  0x090A0B0C, 0x0D0E0F10 };
#define W_INIT2 { 0x71727374, 0x75767778, \
                  0x797A7B7C, 0x7D7E7F70 };
#define D_INIT1 { 0x0102030405060708, \
                  0x090A0B0C0D0E0F10 };
#define D_INIT2 { 0x7172737475767778, \
                  0x797A7B7C7D7E7F70 };

// CHECK-LABEL: define <16 x i8> @test_vpmsumb
vector unsigned char test_vpmsumb(void)
{
  vector unsigned char a = B_INIT1
  vector unsigned char b = B_INIT2
  return __builtin_altivec_crypto_vpmsumb(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vpmsumb
}

// CHECK-LABEL: define <8 x i16> @test_vpmsumh
vector unsigned short test_vpmsumh(void)
{
  vector unsigned short a = H_INIT1
  vector unsigned short b = H_INIT2
  return __builtin_altivec_crypto_vpmsumh(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vpmsumh
}

// CHECK-LABEL: define <4 x i32> @test_vpmsumw
vector unsigned int test_vpmsumw(void)
{
  vector unsigned int a = W_INIT1
  vector unsigned int b = W_INIT2
  return __builtin_altivec_crypto_vpmsumw(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vpmsumw
}

// CHECK-LABEL: define <2 x i64> @test_vpmsumd
vector unsigned long long test_vpmsumd(void)
{
  vector unsigned long long a = D_INIT1
  vector unsigned long long b = D_INIT2
  return __builtin_altivec_crypto_vpmsumd(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vpmsumd
}

// CHECK-LABEL: define <2 x i64> @test_vsbox
vector unsigned long long test_vsbox(void)
{
  vector unsigned long long a = D_INIT1
  return __builtin_altivec_crypto_vsbox(a);
// CHECK: @llvm.ppc.altivec.crypto.vsbox
}

// CHECK-LABEL: define <16 x i8> @test_vpermxorb
vector unsigned char test_vpermxorb(void)
{
  vector unsigned char a = B_INIT1
  vector unsigned char b = B_INIT2
  vector unsigned char c = B_INIT2
  return __builtin_altivec_crypto_vpermxor(a, b, c);
// CHECK: @llvm.ppc.altivec.crypto.vpermxor
}

// CHECK-LABEL: define <8 x i16> @test_vpermxorh
vector unsigned short test_vpermxorh(void)
{
  vector unsigned short a = H_INIT1
  vector unsigned short b = H_INIT2
  vector unsigned short c = H_INIT2
  return __builtin_altivec_crypto_vpermxor(a, b, c);
// CHECK: @llvm.ppc.altivec.crypto.vpermxor
}

// CHECK-LABEL: define <4 x i32> @test_vpermxorw
vector unsigned int test_vpermxorw(void)
{
  vector unsigned int a = W_INIT1
  vector unsigned int b = W_INIT2
  vector unsigned int c = W_INIT2
  return __builtin_altivec_crypto_vpermxor(a, b, c);
// CHECK: @llvm.ppc.altivec.crypto.vpermxor
}

// CHECK-LABEL: define <2 x i64> @test_vpermxord
vector unsigned long long test_vpermxord(void)
{
  vector unsigned long long a = D_INIT1
  vector unsigned long long b = D_INIT2
  vector unsigned long long c = D_INIT2
  return __builtin_altivec_crypto_vpermxor(a, b, c);
// CHECK: @llvm.ppc.altivec.crypto.vpermxor
}

// CHECK-LABEL: test_vpermxorbc
vector bool char test_vpermxorbc(vector bool char a,
                                vector bool char b,
                                vector bool char c) {
  return vec_permxor(a, b, c);
// CHECK: @llvm.ppc.altivec.crypto.vpermxor
}

// CHECK-LABEL: test_vpermxorsc
vector signed char test_vpermxorsc(vector signed char a,
                                   vector signed char b,
                                   vector signed char c) {
  return vec_permxor(a, b, c);
// CHECK: @llvm.ppc.altivec.crypto.vpermxor
}

// CHECK-LABEL: test_vpermxoruc
vector unsigned char test_vpermxoruc(vector unsigned char a,
                                     vector unsigned char b,
                                     vector unsigned char c) {
  return vec_permxor(a, b, c);
// CHECK: @llvm.ppc.altivec.crypto.vpermxor
}

// CHECK-LABEL: define <2 x i64> @test_vcipher
vector unsigned long long test_vcipher(void)
{
  vector unsigned long long a = D_INIT1
  vector unsigned long long b = D_INIT2
  return __builtin_altivec_crypto_vcipher(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vcipher
}

// CHECK-LABEL: define <2 x i64> @test_vcipherlast
vector unsigned long long test_vcipherlast(void)
{
  vector unsigned long long a = D_INIT1
  vector unsigned long long b = D_INIT2
  return __builtin_altivec_crypto_vcipherlast(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vcipherlast
}

// CHECK-LABEL: @test_vncipher
vector unsigned long long test_vncipher(void)
{
  vector unsigned long long a = D_INIT1
  vector unsigned long long b = D_INIT2
  return __builtin_altivec_crypto_vncipher(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vncipher
}

// CHECK-LABEL: define <2 x i64> @test_vncipherlast
vector unsigned long long test_vncipherlast(void)
{
  vector unsigned long long a = D_INIT1
  vector unsigned long long b = D_INIT2
  return __builtin_altivec_crypto_vncipherlast(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vncipherlast
}

// CHECK-LABEL: define <4 x i32> @test_vshasigmaw
vector unsigned int test_vshasigmaw(void)
{
  vector unsigned int a = W_INIT1
  return __builtin_altivec_crypto_vshasigmaw(a, 1, 15);
// CHECK: @llvm.ppc.altivec.crypto.vshasigmaw
}

// CHECK-LABEL: define <2 x i64> @test_vshasigmad
vector unsigned long long test_vshasigmad(void)
{
  vector unsigned long long a = D_INIT2
  return __builtin_altivec_crypto_vshasigmad(a, 1, 15);
// CHECK: @llvm.ppc.altivec.crypto.vshasigmad
}

// Test cases for the builtins the way they are exposed to
// users through altivec.h
// CHECK-LABEL: define <16 x i8> @test_vpmsumb_e
vector unsigned char test_vpmsumb_e(void)
{
  vector unsigned char a = B_INIT1
  vector unsigned char b = B_INIT2
  return __builtin_crypto_vpmsumb(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vpmsumb
}

// CHECK-LABEL: define <8 x i16> @test_vpmsumh_e
vector unsigned short test_vpmsumh_e(void)
{
  vector unsigned short a = H_INIT1
  vector unsigned short b = H_INIT2
  return __builtin_crypto_vpmsumb(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vpmsumh
}

// CHECK-LABEL: define <4 x i32> @test_vpmsumw_e
vector unsigned int test_vpmsumw_e(void)
{
  vector unsigned int a = W_INIT1
  vector unsigned int b = W_INIT2
  return __builtin_crypto_vpmsumb(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vpmsumw
}

// CHECK-LABEL: define <2 x i64> @test_vpmsumd_e
vector unsigned long long test_vpmsumd_e(void)
{
  vector unsigned long long a = D_INIT1
  vector unsigned long long b = D_INIT2
  return __builtin_crypto_vpmsumb(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vpmsumd
}

// CHECK-LABEL: define <2 x i64> @test_vsbox_e
vector unsigned long long test_vsbox_e(void)
{
  vector unsigned long long a = D_INIT1
  return __builtin_crypto_vsbox(a);
// CHECK: @llvm.ppc.altivec.crypto.vsbox
}

// CHECK-LABEL: define <16 x i8> @test_vpermxorb_e
vector unsigned char test_vpermxorb_e(void)
{
  vector unsigned char a = B_INIT1
  vector unsigned char b = B_INIT2
  vector unsigned char c = B_INIT2
  return __builtin_crypto_vpermxor(a, b, c);
// CHECK: @llvm.ppc.altivec.crypto.vpermxor
}

// CHECK-LABEL: define <8 x i16> @test_vpermxorh_e
vector unsigned short test_vpermxorh_e(void)
{
  vector unsigned short a = H_INIT1
  vector unsigned short b = H_INIT2
  vector unsigned short c = H_INIT2
  return __builtin_crypto_vpermxor(a, b, c);
// CHECK: @llvm.ppc.altivec.crypto.vpermxor
}

// CHECK-LABEL: define <4 x i32> @test_vpermxorw_e
vector unsigned int test_vpermxorw_e(void)
{
  vector unsigned int a = W_INIT1
  vector unsigned int b = W_INIT2
  vector unsigned int c = W_INIT2
  return __builtin_crypto_vpermxor(a, b, c);
// CHECK: @llvm.ppc.altivec.crypto.vpermxor
}

// CHECK-LABEL: define <2 x i64> @test_vpermxord_e
vector unsigned long long test_vpermxord_e(void)
{
  vector unsigned long long a = D_INIT1
  vector unsigned long long b = D_INIT2
  vector unsigned long long c = D_INIT2
  return __builtin_crypto_vpermxor(a, b, c);
// CHECK: @llvm.ppc.altivec.crypto.vpermxor
}

// CHECK-LABEL: define <2 x i64> @test_vcipher_e
vector unsigned long long test_vcipher_e(void)
{
  vector unsigned long long a = D_INIT1
  vector unsigned long long b = D_INIT2
  return __builtin_crypto_vcipher(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vcipher
}

// CHECK-LABEL: define <2 x i64> @test_vcipherlast_e
vector unsigned long long test_vcipherlast_e(void)
{
  vector unsigned long long a = D_INIT1
  vector unsigned long long b = D_INIT2
  return __builtin_crypto_vcipherlast(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vcipherlast
}

// CHECK-LABEL: define <2 x i64> @test_vncipher_e
vector unsigned long long test_vncipher_e(void)
{
  vector unsigned long long a = D_INIT1
  vector unsigned long long b = D_INIT2
  return __builtin_crypto_vncipher(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vncipher
}

// CHECK-LABEL: define <2 x i64> @test_vncipherlast_e
vector unsigned long long test_vncipherlast_e(void)
{
  vector unsigned long long a = D_INIT1
  vector unsigned long long b = D_INIT2
  return __builtin_crypto_vncipherlast(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vncipherlast
}

// CHECK-LABEL: define <4 x i32> @test_vshasigmaw_e
vector unsigned int test_vshasigmaw_e(void)
{
  vector unsigned int a = W_INIT1
  return __builtin_crypto_vshasigmaw(a, 1, 15);
// CHECK: @llvm.ppc.altivec.crypto.vshasigmaw
}

// CHECK-LABEL: define <2 x i64> @test_vshasigmad_e
vector unsigned long long test_vshasigmad_e(void)
{
  vector unsigned long long a = D_INIT2
  return __builtin_crypto_vshasigmad(a, 0, 15);
// CHECK: @llvm.ppc.altivec.crypto.vshasigmad
}

// CHECK-LABEL: @test_vec_sbox_be
vector unsigned char test_vec_sbox_be(void)
{
  vector unsigned char a = B_INIT1
  return vec_sbox_be(a);
// CHECK: @llvm.ppc.altivec.crypto.vsbox
}

// CHECK-LABEL: @test_vec_cipher_be
vector unsigned char test_vec_cipher_be(void)
{
  vector unsigned char a = B_INIT1
  vector unsigned char b = B_INIT2
  return vec_cipher_be(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vcipher
}

// CHECK-LABEL: @test_vec_cipherlast_be
vector unsigned char test_vec_cipherlast_be(void)
{
  vector unsigned char a = B_INIT1
  vector unsigned char b = B_INIT2
  return vec_cipherlast_be(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vcipherlast
}

// CHECK-LABEL: @test_vec_ncipher_be
vector unsigned char test_vec_ncipher_be(void)
{
  vector unsigned char a = B_INIT1
  vector unsigned char b = B_INIT2
  return vec_ncipher_be(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vncipher
}

// CHECK-LABEL: @test_vec_ncipherlast_be
vector unsigned char test_vec_ncipherlast_be(void)
{
  vector unsigned char a = B_INIT1
  vector unsigned char b = B_INIT2
  return vec_ncipherlast_be(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vncipherlast
}

// CHECK-LABEL: @test_vec_shasigma_bew
vector unsigned int test_vec_shasigma_bew(void)
{
  vector unsigned int a = W_INIT1
  return vec_shasigma_be(a, 1, 15);
// CHECK: @llvm.ppc.altivec.crypto.vshasigmaw
}

// CHECK-LABEL: @test_vec_shasigma_bed
vector unsigned long long test_vec_shasigma_bed(void)
{
  vector unsigned long long a = D_INIT2
  return vec_shasigma_be(a, 1, 15);
// CHECK: @llvm.ppc.altivec.crypto.vshasigmad
}

// CHECK-LABEL: @test_vec_pmsum_beb
vector unsigned short test_vec_pmsum_beb(void)
{
  vector unsigned char a = B_INIT1
  vector unsigned char b = B_INIT2
  return vec_pmsum_be(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vpmsumb
}

// CHECK-LABEL: @test_vec_pmsum_beh
vector unsigned int test_vec_pmsum_beh(void)
{
  vector unsigned short a = H_INIT1
  vector unsigned short b = H_INIT2
  return vec_pmsum_be(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vpmsumh
}

// CHECK-LABEL: @test_vec_pmsum_bew
vector unsigned long long test_vec_pmsum_bew(void)
{
  vector unsigned int a = W_INIT1
  vector unsigned int b = W_INIT2
  return vec_pmsum_be(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vpmsumw
}

// CHECK-LABEL: @test_vec_pmsum_bed
vector unsigned __int128 test_vec_pmsum_bed(void)
{
  vector unsigned long long a = D_INIT1
  vector unsigned long long b = D_INIT2
  return vec_pmsum_be(a, b);
// CHECK: @llvm.ppc.altivec.crypto.vpmsumd
}

