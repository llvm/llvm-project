// Make sure that OpenCL builtins declared in opencl.h are translated
// to corresponding SPIR-V instructions, not a function calls.
// Builtins must be declared as overloadable, so Clang mangles their names,
// and LLVM-SPIRV translator can recognize them.

// RUN: %clang_cc1 %s -emit-llvm-bc -triple spir-unknown-unknown -O0 -cl-std=CL2.0 -include opencl-c.h -o %t.bc
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s

// CHECK: CreateUserEvent
// CHECK: IsValidEvent
// CHECK-NOT: FunctionCall

bool test() {
  clk_event_t e = create_user_event();
  return is_valid_event(e);
}
