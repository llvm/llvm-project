// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 2>&1 | FileCheck %s --check-prefix=CHECK-32-ERROR

extern unsigned long long ull;
extern unsigned long long *ull_addr;

// CHECK-LABEL: @test_builtin_ppc_store8r(
// CHECK:         [[TMP0:%.*]] = load i64, ptr @ull, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load ptr, ptr @ull_addr, align 8
// CHECK-NEXT:    call void @llvm.ppc.store8r(i64 [[TMP0]], ptr [[TMP1]])
// CHECK-NEXT:    ret void

// CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
void test_builtin_ppc_store8r() {
  __builtin_ppc_store8r(ull, ull_addr);
}

// CHECK-LABEL: @test_builtin_ppc_load8r(
// CHECK:         [[TMP0:%.*]] = load ptr, ptr @ull_addr, align 8
// CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.ppc.load8r(ptr [[TMP0]])
// CHECK-NEXT:    ret i64 [[TMP2]]

// CHECK-32-ERROR: error: this builtin is only available on 64-bit targets
unsigned long long test_builtin_ppc_load8r() {
  return __builtin_ppc_load8r(ull_addr);
}
