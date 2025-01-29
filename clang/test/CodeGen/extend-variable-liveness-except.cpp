// RUN: %clang_cc1 %s -triple %itanium_abi_triple -emit-llvm -fextend-variable-liveness -fcxx-exceptions -fexceptions -o - | FileCheck %s
// This test checks that the fake uses can be generated in exception handling
// blocks and that we can emit fake uses for the __int128 data type.

extern int bar();

/// Try block: fake use ends at try-block scope.
// [[BAR_VAL::%[a-zA-Z0-9\.]+]] = invoke{{.*}} i32 @_Z3barv()
// store i32 %[[BAR_VAL]], ptr [[K_ALLOC_VAL:%[a-zA-Z0-9\.]+]], align 4
// [[K_FAKE_USE:%[a-zA-Z0-9\.]+]] = load i32, ptr [[K_ALLOC_VAL]], align 4
// call void (...) @llvm.fake.use(i32 [[K_FAKE_USE]]) #2
// br label

/// Catch block: fetching the caught value...
// CHECK: [[CATCH_PTR:%[a-zA-Z0-9\.]+]] = call ptr @__cxa_begin_catch(
// CHECK: [[L_VAL:%[a-zA-Z0-9\.]+]] = load i32, ptr [[CATCH_PTR]], align 4

/// Storing to allocas...
// CHECK-DAG: store i32 8, ptr [[M_ALLOC_VAL:%[a-zA-Z0-9\.]+]]
// CHECK-DAG: store i32 [[L_VAL]], ptr [[L_ALLOC_VAL:%[a-zA-Z0-9\.]+]], align 4

/// Load into fake uses - expect M to precede L.
// CHECK: [[M_FAKE_VAL:%[a-zA-Z0-9\.]+]] = load i32, ptr [[M_ALLOC_VAL]]
// CHECK: call void (...) @llvm.fake.use(i32 [[M_FAKE_VAL]])
// CHECK: [[L_FAKE_VAL:%[a-zA-Z0-9\.]+]] = load i32, ptr [[L_ALLOC_VAL]]
// CHECK: call void (...) @llvm.fake.use(i32 [[L_FAKE_VAL]])
void foo() {
  try {
    int k = bar();
  } catch (int l) {
    /// The catch block contains a fake use for the local within its scope.
    int m = 8;
  }
}
