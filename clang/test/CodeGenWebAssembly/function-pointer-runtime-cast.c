// REQUIRES: webassembly-registered-target
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -emit-llvm -O0 -fwasm-fix-function-bitcasts -o - %s | FileCheck %s

// Test runtime function pointer cast with different argument counts
// This simulates cases like g_list_free_full where a function pointer parameter
// is cast from fewer params to more params

typedef void (*OneArgFunc)(void *);
typedef void (*TwoArgFunc)(void *, void *);

// Check for both TLS globals at the top of the output
// CHECK: @__wasm_runtime_wrapper_vi_to_vii_fptr = linkonce_odr thread_local global ptr null
// CHECK: @__wasm_runtime_wrapper_iii_to_vii_fptr = linkonce_odr thread_local global ptr null

// A function with one argument
void my_one_arg_func(void *ptr) {
  // Do something
}

// Test case 1: Direct call of casted runtime function pointer
// CHECK-LABEL: @runtime_cast_caller
void runtime_cast_caller(OneArgFunc fp, void *data) {
  // Cast the runtime parameter from 1-arg to 2-arg signature and call directly
  // CHECK: store ptr %{{.*}}, ptr @__wasm_runtime_wrapper_vi_to_vii_fptr
  // CHECK: call void @__wasm_runtime_wrapper_vi_to_vii(ptr
  ((TwoArgFunc)fp)(data, (void*)0);
}

// The runtime wrapper should be generated once and shared by both cases
// CHECK-LABEL: define linkonce_odr void @__wasm_runtime_wrapper_vi_to_vii(ptr %0, ptr %1)
// CHECK: %{{.*}} = load ptr, ptr @__wasm_runtime_wrapper_vi_to_vii_fptr
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
  // CHECK: store ptr %{{.*}}, ptr @__wasm_runtime_wrapper_vi_to_vii_fptr
  // CHECK: call void @library_function(ptr noundef @__wasm_runtime_wrapper_vi_to_vii
  library_function((TwoArgFunc)fp, data);
}

// A function with two arguments that returns int (like a comparator)
int my_compare_func(void *a, void *b) {
  return 0;
}

// Test case 3: Same param count, return type coercion (int -> void)
// This simulates g_slist_sort where int compare(void*, void*) is cast to void func(void*, void*)
// Use a typedef to create a function pointer type with int return
typedef int (*CompareFunc)(void *, void *);

// CHECK-LABEL: @test_return_coercion
void test_return_coercion(CompareFunc fp, void *data) {
  // Cast int(void*, void*) -> void(void*, void*) on a runtime parameter
  // CHECK: store ptr %{{.*}}, ptr @__wasm_runtime_wrapper_iii_to_vii_fptr
  // CHECK: call void @library_function(ptr noundef @__wasm_runtime_wrapper_iii_to_vii
  library_function((TwoArgFunc)fp, data);
}

// The runtime wrapper for int->void coercion with same param count
// CHECK-LABEL: define linkonce_odr void @__wasm_runtime_wrapper_iii_to_vii(ptr %0, ptr %1)
// CHECK: %{{.*}} = load ptr, ptr @__wasm_runtime_wrapper_iii_to_vii_fptr
// CHECK: %{{.*}} = call i32 %{{.*}}(ptr %0, ptr %1)
// CHECK: ret void

// CHECK-LABEL: @test
void test() {
  // Test both scenarios
  runtime_cast_caller(my_one_arg_func, (void*)0);
  indirect_caller(my_one_arg_func, (void*)0);
  test_return_coercion(my_compare_func, (void*)0);
}
