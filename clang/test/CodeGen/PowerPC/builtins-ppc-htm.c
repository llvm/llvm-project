// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +altivec -target-feature +htm -triple powerpc64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: not %clang_cc1 -target-feature +altivec -target-feature -htm -triple powerpc64-unknown-unknown -emit-llvm-only %s 2>&1 | FileCheck %s --check-prefix=ERROR

void test1(long int *r, int code, long int *a, long int *b) {
// CHECK-LABEL: define{{.*}} void @test1

  r[0] = __builtin_tbegin (0);
// CHECK: @llvm.ppc.tbegin
// ERROR: error: '__builtin_tbegin' needs target feature htm
  r[1] = __builtin_tbegin (1);
// CHECK: @llvm.ppc.tbegin
// ERROR: error: '__builtin_tbegin' needs target feature htm
  r[2] = __builtin_tend (0);
// CHECK: @llvm.ppc.tend
// ERROR: error: '__builtin_tend' needs target feature htm
  r[3] = __builtin_tendall ();
// CHECK: @llvm.ppc.tendall
// ERROR: error: '__builtin_tendall' needs target feature htm

  r[4] = __builtin_tabort (code);
// CHECK: @llvm.ppc.tabort
// ERROR: error: '__builtin_tabort' needs target feature htm
  r[5] = __builtin_tabort (0x1);
// CHECK: @llvm.ppc.tabort
// ERROR: error: '__builtin_tabort' needs target feature htm
  r[6] = __builtin_tabortdc (0xf, a[0], b[0]);
// CHECK: @llvm.ppc.tabortdc
// ERROR: error: '__builtin_tabortdc' needs target feature htm
  r[7] = __builtin_tabortdci (0xf, a[1], 0x1);
// CHECK: @llvm.ppc.tabortdc
// ERROR: error: '__builtin_tabortdci' needs target feature htm
  r[8] = __builtin_tabortwc (0xf, a[2], b[2]);
// CHECK: @llvm.ppc.tabortwc
// ERROR: error: '__builtin_tabortwc' needs target feature htm
  r[9] = __builtin_tabortwci (0xf, a[3], 0x1);
// CHECK: @llvm.ppc.tabortwc
// ERROR: error: '__builtin_tabortwci' needs target feature htm

  r[10] = __builtin_tcheck ();
// CHECK: @llvm.ppc.tcheck
// ERROR: error: '__builtin_tcheck' needs target feature htm
  r[11] = __builtin_trechkpt ();
// CHECK: @llvm.ppc.trechkpt
// ERROR: error: '__builtin_trechkpt' needs target feature htm
  r[12] = __builtin_treclaim (0);
// CHECK: @llvm.ppc.treclaim
// ERROR: error: '__builtin_treclaim' needs target feature htm
  r[13] = __builtin_tresume ();
// CHECK: @llvm.ppc.tresume
// ERROR: error: '__builtin_tresume' needs target feature htm
  r[14] = __builtin_tsuspend ();
// CHECK: @llvm.ppc.tsuspend
// ERROR: error: '__builtin_tsuspend' needs target feature htm
  r[15] = __builtin_tsr (0);
// CHECK: @llvm.ppc.tsr
// ERROR: error: '__builtin_tsr' needs target feature htm

  r[16] = __builtin_ttest ();
// CHECK: @llvm.ppc.ttest
// ERROR: error: '__builtin_ttest' needs target feature htm

  r[17] = __builtin_get_texasr ();
// CHECK: @llvm.ppc.get.texasr
// ERROR: error: '__builtin_get_texasr' needs target feature htm
  r[18] = __builtin_get_texasru ();
// CHECK: @llvm.ppc.get.texasru
// ERROR: error: '__builtin_get_texasru' needs target feature htm
  r[19] = __builtin_get_tfhar ();
// CHECK: @llvm.ppc.get.tfhar
// ERROR: error: '__builtin_get_tfhar' needs target feature htm
  r[20] = __builtin_get_tfiar ();
// CHECK: @llvm.ppc.get.tfiar
// ERROR: error: '__builtin_get_tfiar' needs target feature htm

  __builtin_set_texasr (a[21]);
// CHECK: @llvm.ppc.set.texasr
// ERROR: error: '__builtin_set_texasr' needs target feature htm
  __builtin_set_texasru (a[22]);
// CHECK: @llvm.ppc.set.texasru
// ERROR: error: '__builtin_set_texasru' needs target feature htm
  __builtin_set_tfhar (a[23]);
// CHECK: @llvm.ppc.set.tfhar
// ERROR: error: '__builtin_set_tfhar' needs target feature htm
  __builtin_set_tfiar (a[24]);
// CHECK: @llvm.ppc.set.tfiar
// ERROR: error: '__builtin_set_tfiar' needs target feature htm
}
