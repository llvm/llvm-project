// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// Test that __ocl_event_t is available in SYCL device code and lowers to
// target("spirv.Event") in LLVM IR for SPIR-V targets.

[[clang::sycl_external]] void test_ocl_event_param(__ocl_event_t evt) {}
// CHECK: define{{.*}} void @_Z20test_ocl_event_param9ocl_event(target("spirv.Event") %evt)

[[clang::sycl_external]] void test_ocl_event() {
  __ocl_event_t evt;
  // CHECK: %evt = alloca target("spirv.Event")
  test_ocl_event_param(evt);
  // CHECK: call spir_func void @_Z20test_ocl_event_param9ocl_event(target("spirv.Event") %0)
}
