// Make sure that OpenCL builtins declared in opencl.h are translated
// to corresponding SPIR-V instructions, not a function calls.
// Builtins must be declared as overloadable, so Clang mangles their names,
// and LLVM-SPIRV translator can recognize them.

// RUN: %clang_cc1 %s -emit-llvm -triple spir-unknown-unknown -O0 -cl-std=CL2.0 -include opencl-c.h -o - | FileCheck %s

// CHECK: %[[CALL_USEREVENT:[a-z0-9]+]] = call spir_func %opencl.clk_event_t* @_Z17create_user_eventv()
// CHECK: store %opencl.clk_event_t* %[[CALL_USEREVENT]], %opencl.clk_event_t** %e, align 4
// CHECK: %[[VAR_EVENTPTR:[a-z0-9]+]] = load %opencl.clk_event_t*, %opencl.clk_event_t** %e, align 4
// CHECK: %[[CALL_ISVALIDEVENT:[a-z0-9]+]] = call spir_func zeroext i1 @_Z14is_valid_event12ocl_clkevent(%opencl.clk_event_t* %[[VAR_EVENTPTR]])
// CHECK: ret i1 %[[CALL_ISVALIDEVENT]]

bool test() {
  clk_event_t e = create_user_event();
  return is_valid_event(e);
}
