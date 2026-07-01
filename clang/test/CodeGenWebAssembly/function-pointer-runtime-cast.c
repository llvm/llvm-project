// REQUIRES: webassembly-registered-target
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -emit-llvm -O0 -fwasm-fix-function-bitcasts -o - %s | FileCheck %s

// Test runtime function pointer cast with different argument counts
// This simulates cases like g_list_free_full where a function pointer parameter
// is cast from fewer params to more params

typedef void (*OneArgFunc)(void *);
typedef void (*TwoArgFunc)(void *, void *);

// Pool counters
// CHECK: @__wasm_runtime_pool___wasm_runtime_wrapper_vi_to_vii{{.*}}_counter = internal thread_local global i32 0
// CHECK: @__wasm_runtime_pool___wasm_runtime_wrapper_iii_to_vii{{.*}}_counter = internal thread_local global i32 0

// A function with one argument
void my_one_arg_func(void *ptr) {
  // Do something
}

// Test case 1: Direct call of casted runtime function pointer
// CHECK-LABEL: @runtime_cast_caller
void runtime_cast_caller(OneArgFunc fp, void *data) {
  // CHECK: atomicrmw add ptr @__wasm_runtime_pool___wasm_runtime_wrapper_vi_to_vii{{.*}}_counter, i32 1
  // CHECK: store ptr %{{.*}}, ptr %
  // CHECK: load ptr, ptr %
  ((TwoArgFunc)fp)(data, (void*)0);
}

// Pool wrapper functions (internal linkage, one per slot)
// CHECK-LABEL: define internal void @__wasm_runtime_wrapper_vi_to_vii{{.*}}_0(ptr %0, ptr %1)
// CHECK: load ptr, ptr
// CHECK: call void %{{.*}}(ptr %0)
// CHECK: ret void

// CHECK-LABEL: define internal void @__wasm_runtime_wrapper_vi_to_vii{{.*}}_1(ptr %0, ptr %1)
// CHECK: load ptr, ptr
// CHECK: call void %{{.*}}(ptr %0)
// CHECK: ret void

// Test case 2: Pass casted runtime function pointer to another function
// This is closer to the real g_list_free_full scenario
// CHECK-LABEL: @library_function
void library_function(TwoArgFunc func, void *data) {
  // CHECK: call void %{{.*}}(ptr noundef %{{.*}}, ptr noundef null)
  func(data, (void*)0);
}

// CHECK-LABEL: @indirect_caller
void indirect_caller(OneArgFunc fp, void *data) {
  // Cast and pass to another function (like g_list_free_full does)
  // CHECK: atomicrmw add ptr @__wasm_runtime_pool___wasm_runtime_wrapper_vi_to_vii{{.*}}_counter, i32 1
  library_function((TwoArgFunc)fp, data);
}

// A function with two arguments that returns int (like a comparator)
int my_compare_func(void *a, void *b) {
  return 0;
}

// Test case 3: Same param count, return type coercion (int -> void)
typedef int (*CompareFunc)(void *, void *);

// CHECK-LABEL: @test_return_coercion
void test_return_coercion(void *opaque_fp, void *data) {
  // Load through void* to prevent static thunk tracing
  // CHECK: atomicrmw add ptr @__wasm_runtime_pool___wasm_runtime_wrapper_iii_to_vii{{.*}}_counter, i32 1
  CompareFunc fp = (CompareFunc)(__typeof__(CompareFunc))opaque_fp;
  library_function((TwoArgFunc)fp, data);
}

// Pool wrapper for int->void coercion
// CHECK-LABEL: define internal void @__wasm_runtime_wrapper_iii_to_vii{{.*}}_0(ptr %0, ptr %1)
// CHECK: load ptr, ptr
// CHECK: call i32 %{{.*}}(ptr %0, ptr %1)
// CHECK: ret void

// Store a casted function pointer for later. This simulates the closure
// pattern where multiple closures store different marshals via the same
// cast expression.
TwoArgFunc saved1;
TwoArgFunc saved2;

// CHECK-LABEL: @test_store_for_later
void test_store_for_later(OneArgFunc fp1, OneArgFunc fp2) {
  // Each store claims a different pool slot
  // CHECK: atomicrmw add ptr @__wasm_runtime_pool___wasm_runtime_wrapper_vi_to_vii{{.*}}_counter, i32 1
  saved1 = (TwoArgFunc)fp1;
  // CHECK: atomicrmw add ptr @__wasm_runtime_pool___wasm_runtime_wrapper_vi_to_vii{{.*}}_counter, i32 1
  saved2 = (TwoArgFunc)fp2;
}

// CHECK-LABEL: @test
void test() {
  runtime_cast_caller(my_one_arg_func, (void*)0);
  indirect_caller(my_one_arg_func, (void*)0);
  test_return_coercion((void*)my_compare_func, (void*)0);
  test_store_for_later(my_one_arg_func, my_one_arg_func);
}
