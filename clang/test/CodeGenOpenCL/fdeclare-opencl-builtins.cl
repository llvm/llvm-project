// RUN: %clang_cc1 -emit-llvm -o - -O0 -triple spir-unknown-unknown -cl-std=CL1.2 -finclude-default-header %s \
// RUN: | FileCheck %s --check-prefixes CHECK,CHECK-NOGAS
// RUN: %clang_cc1 -emit-llvm -o - -O0 -triple spir-unknown-unknown -cl-std=CL1.2 -fdeclare-opencl-builtins -finclude-default-header %s \
// RUN: | FileCheck %s --check-prefixes CHECK,CHECK-NOGAS
// RUN: %clang_cc1 -emit-llvm -o - -O0 -triple spir-unknown-unknown -cl-std=CL3.0 -fdeclare-opencl-builtins -finclude-default-header %s \
// RUN: | FileCheck %s --check-prefixes CHECK,CHECK-GAS
// RUN: %clang_cc1 -emit-llvm -o - -O0 -triple spir-unknown-unknown -cl-std=CL3.0 -fdeclare-opencl-builtins -finclude-default-header \
// RUN: -cl-ext=-__opencl_c_generic_address_space,-__opencl_c_pipes,-__opencl_c_device_enqueue %s \
// RUN: | FileCheck %s --check-prefixes CHECK,CHECK-NOGAS

// Test that mix is correctly defined.
// CHECK-LABEL: @test_float
// CHECK: call spir_func <4 x float> @_Z3mixDv4_fS_f
// CHECK: ret
void test_float(float4 x, float a) {
  float4 ret = mix(x, x, a);
}

// Test that Attr.Const from OpenCLBuiltins.td is lowered to a readnone attribute.
// CHECK-LABEL: @test_const_attr
// CHECK: call spir_func i32 @_Z3maxii({{.*}}) [[ATTR_CONST:#[0-9]]]
// CHECK: ret
int test_const_attr(int a) {
  return max(a, 2);
}

// Test that Attr.Pure from OpenCLBuiltins.td is lowered to a readonly attribute.
// CHECK-LABEL: @test_pure_attr
// CHECK: call spir_func <4 x float> @_Z11read_imagef{{.*}} [[ATTR_PURE:#[0-9]]]
// CHECK: ret
kernel void test_pure_attr(read_only image1d_t img) {
  float4 resf = read_imagef(img, 42);
}

// Test that builtins with only one prototype are mangled.
// CHECK-LABEL: @test_mangling
// CHECK: call spir_func i32 @_Z12get_local_idj
kernel void test_mangling() {
  size_t lid = get_local_id(0);
}

// Test that the correct builtin is called depending on the generic address
// space feature availability.
// CHECK-LABEL: @test_generic_optionality
// CHECK-GAS: call spir_func float @_Z5fractfPU3AS4f
// CHECK-NOGAS: call spir_func float @_Z5fractfPf
void test_generic_optionality(float a, float *b) {
  float res = fract(a, b);
}

// CHECK: attributes [[ATTR_CONST]] =
// CHECK-SAME: memory(none)
// CHECK: attributes [[ATTR_PURE]] =
// CHECK-SAME: memory(read)
