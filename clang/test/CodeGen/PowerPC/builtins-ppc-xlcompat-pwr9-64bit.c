// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm %s \
// RUN:   -target-cpu pwr9 -o - | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc-unknown-aix -emit-llvm-only %s \
// RUN:   -target-cpu pwr9 2>&1 | FileCheck %s --check-prefix=CHECK-32-ERROR
// RUN: not %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm-only %s \
// RUN:   -target-cpu pwr8 2>&1 | FileCheck %s --check-prefix=CHECK-NONPWR9-ERR

extern signed long long sll;
extern unsigned long long ull;
double d;

void test_compat_builtins() {
  // CHECK-LABEL: @test_compat_builtins(
  // CHECK: %2 = call i64 @llvm.ppc.cmpeqb(i64 %0, i64 %1)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR: error: '__builtin_ppc_cmpeqb' needs target feature isa-v30-instructions
  sll = __builtin_ppc_cmpeqb(sll, sll);

  // CHECK: %5 = call i64 @llvm.ppc.setb(i64 %3, i64 %4)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR: error: '__builtin_ppc_setb' needs target feature isa-v30-instructions
  sll = __builtin_ppc_setb(sll, sll);

  // CHECK: %9 = call i64 @llvm.ppc.maddhd(i64 %6, i64 %7, i64 %8)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR: error: '__builtin_ppc_maddhd' needs target feature isa-v30-instructions
  sll = __builtin_ppc_maddhd(sll, sll, sll);

  // CHECK: %13 = call i64 @llvm.ppc.maddhdu(i64 %10, i64 %11, i64 %12)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR: error: '__builtin_ppc_maddhdu' needs target feature isa-v30-instructions
  ull = __builtin_ppc_maddhdu(ull, ull, ull);

  // CHECK: %17 = call i64 @llvm.ppc.maddld(i64 %14, i64 %15, i64 %16)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR: error: '__builtin_ppc_maddld' needs target feature isa-v30-instructions
  sll = __builtin_ppc_maddld(sll, sll, sll);

  // CHECK: %21 = call i64 @llvm.ppc.maddld(i64 %18, i64 %19, i64 %20)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR: error: '__builtin_ppc_maddld' needs target feature isa-v30-instructions
  ull = __builtin_ppc_maddld(ull, ull, ull);

  // CHECK: %23 = call i64 @llvm.ppc.extract.sig(double %22)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR: error: '__builtin_ppc_extract_sig' needs target feature power9-vector
  ull = __extract_sig (d);

  // CHECK: %26 = call double @llvm.ppc.insert.exp(double %24, i64 %25)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR: error: '__builtin_ppc_insert_exp' needs target feature power9-vector
  d = __insert_exp (d, ull);

  // CHECK: %29 = call i64 @llvm.ppc.addex(i64 %27, i64 %28, i32 0)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR: error: '__builtin_ppc_addex' needs target feature isa-v30-instructions
  sll = __builtin_ppc_addex(sll, sll, 0);

  // CHECK: %32 = call i64 @llvm.ppc.addex(i64 %30, i64 %31, i32 0)
  // CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
  // CHECK-NONPWR9-ERR: error: '__builtin_ppc_addex' needs target feature isa-v30-instructions
  ull = __builtin_ppc_addex(ull, ull, 0);
}
