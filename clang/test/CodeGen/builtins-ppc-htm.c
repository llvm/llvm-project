// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +altivec -target-feature +htm -triple powerpc64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

void test1(long int *r, int code, long int *a, long int *b) {
// CHECK-LABEL: define void @test1

  r[0] = __builtin_tbegin (0);
// CHECK: @llvm.ppc.tbegin
  r[1] = __builtin_tbegin (1);
// CHECK: @llvm.ppc.tbegin
  r[2] = __builtin_tend (0);
// CHECK: @llvm.ppc.tend
  r[3] = __builtin_tendall ();
// CHECK: @llvm.ppc.tendall

  r[4] = __builtin_tabort (code);
// CHECK: @llvm.ppc.tabort
  r[5] = __builtin_tabort (0x1);
// CHECK: @llvm.ppc.tabort
  r[6] = __builtin_tabortdc (0xf, a[0], b[0]);
// CHECK: @llvm.ppc.tabortdc
  r[7] = __builtin_tabortdci (0xf, a[1], 0x1);
// CHECK: @llvm.ppc.tabortdc
  r[8] = __builtin_tabortwc (0xf, a[2], b[2]);
// CHECK: @llvm.ppc.tabortwc
  r[9] = __builtin_tabortwci (0xf, a[3], 0x1);
// CHECK: @llvm.ppc.tabortwc

  r[10] = __builtin_tcheck ();
// CHECK: @llvm.ppc.tcheck
  r[11] = __builtin_trechkpt ();
// CHECK: @llvm.ppc.trechkpt
  r[12] = __builtin_treclaim (0);
// CHECK: @llvm.ppc.treclaim
  r[13] = __builtin_tresume ();
// CHECK: @llvm.ppc.tresume
  r[14] = __builtin_tsuspend ();
// CHECK: @llvm.ppc.tsuspend
  r[15] = __builtin_tsr (0);
// CHECK: @llvm.ppc.tsr

  r[16] = __builtin_ttest ();
// CHECK: @llvm.ppc.ttest

  r[17] = __builtin_get_texasr ();
// CHECK: @llvm.ppc.get.texasr
  r[18] = __builtin_get_texasru ();
// CHECK: @llvm.ppc.get.texasru
  r[19] = __builtin_get_tfhar ();
// CHECK: @llvm.ppc.get.tfhar
  r[20] = __builtin_get_tfiar ();
// CHECK: @llvm.ppc.get.tfiar

  __builtin_set_texasr (a[21]);
// CHECK: @llvm.ppc.set.texasr
  __builtin_set_texasru (a[22]);
// CHECK: @llvm.ppc.set.texasru
  __builtin_set_tfhar (a[23]);
// CHECK: @llvm.ppc.set.tfhar
  __builtin_set_tfiar (a[24]);
// CHECK: @llvm.ppc.set.tfiar
}
