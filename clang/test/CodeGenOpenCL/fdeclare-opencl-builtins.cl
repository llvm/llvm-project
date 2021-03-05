// RUN: %clang_cc1 -emit-llvm -o - -O0 -triple spir-unknown-unknown -cl-std=CL1.2 -fdeclare-opencl-builtins -finclude-default-header %s | FileCheck %s

// Test that mix is correctly defined.
// CHECK-LABEL: @test_float
// CHECK: call <4 x float> @_Z3mixDv4_fS_f
// CHECK: ret
void test_float(float4 x, float a) {
  float4 ret = mix(x, x, a);
}

// Test that Attr.Const from OpenCLBuiltins.td is lowered to a readnone attribute.
// CHECK-LABEL: @test_const_attr
// CHECK: call i32 @_Z3maxii({{.*}}) [[ATTR_CONST:#[0-9]]]
// CHECK: ret
int test_const_attr(int a) {
  return max(a, 2);
}

// Test that Attr.Pure from OpenCLBuiltins.td is lowered to a readonly attribute.
// CHECK-LABEL: @test_pure_attr
// CHECK: call <4 x float> @_Z11read_imagef{{.*}} [[ATTR_PURE:#[0-9]]]
// CHECK: ret
kernel void test_pure_attr(read_only image1d_t img) {
  float4 resf = read_imagef(img, 42);
}

// Test that builtins with only one prototype are mangled.
// CHECK-LABEL: @test_mangling
// CHECK: call i32 @_Z12get_local_idj
kernel void test_mangling() {
  size_t lid = get_local_id(0);
}

// CHECK: attributes [[ATTR_CONST]] =
// CHECK-SAME: readnone
// CHECK: attributes [[ATTR_PURE]] =
// CHECK-SAME: readonly
