// RUN: %clang_cc1 %s -cl-std=CL1.2 -emit-llvm -triple x86_64-unknown-unknown -o - | FileCheck %s
// RUN: %clang_cc1 %s -cl-std=CL1.2 -emit-llvm -triple amdgcn-unknown-unknown -o - | FileCheck -check-prefixes=AMDGCN %s
// Test that the kernels always use the SPIR calling convention
// to have unambiguous mapping of arguments to feasibly implement
// clSetKernelArg().

typedef struct int_single {
    int a;
} int_single;

typedef struct int_pair {
    long a;
    long b;
} int_pair;

typedef struct test_struct {
    int elementA;
    int elementB;
    long elementC;
    char elementD;
    long elementE;
    float elementF;
    short elementG;
    double elementH;
} test_struct;

kernel void test_single(int_single input, global int* output) {
// CHECK: spir_kernel
// AMDGCN: define{{.*}} amdgpu_kernel void @test_single
// CHECK: ptr {{.*}} byval(%struct.int_single) align 4 captures(none)
// CHECK: ptr noundef writeonly align 4 captures(none) initializes((0, 4)) %output
 output[0] = input.a;
}

kernel void test_pair(int_pair input, global int* output) {
// CHECK: spir_kernel
// AMDGCN: define{{.*}} amdgpu_kernel void @test_pair
// CHECK: ptr {{.*}} byval(%struct.int_pair) align 8 captures(none)
// CHECK: ptr noundef writeonly align 4 captures(none) initializes((0, 8)) %output
 output[0] = (int)input.a;
 output[1] = (int)input.b;
}

kernel void test_kernel(test_struct input, global int* output) {
// CHECK: spir_kernel
// AMDGCN: define{{.*}} amdgpu_kernel void @test_kernel
// CHECK: ptr {{.*}} byval(%struct.test_struct) align 8 captures(none)
// CHECK: ptr noundef writeonly align 4 captures(none) initializes((0, 32)) %output
 output[0] = input.elementA;
 output[1] = input.elementB;
 output[2] = (int)input.elementC;
 output[3] = (int)input.elementD;
 output[4] = (int)input.elementE;
 output[5] = (int)input.elementF;
 output[6] = (int)input.elementG;
 output[7] = (int)input.elementH;
};

void test_function(int_pair input, global int* output) {
// CHECK-NOT: spir_kernel
// AMDGCN-NOT: define{{.*}} amdgpu_kernel void @test_function
// CHECK: i64 %input.coerce0, i64 %input.coerce1, ptr noundef writeonly captures(none) initializes((0, 8)) %output
 output[0] = (int)input.a;
 output[1] = (int)input.b;
}
