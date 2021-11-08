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
// CHECK: struct.int_single* nocapture {{.*}} byval(%struct.int_single)
// CHECK: i32* nocapture %output
 output[0] = input.a;
}

kernel void test_pair(int_pair input, global int* output) {
// CHECK: spir_kernel
// AMDGCN: define{{.*}} amdgpu_kernel void @test_pair
// CHECK: struct.int_pair* nocapture {{.*}} byval(%struct.int_pair)
// CHECK: i32* nocapture %output
 output[0] = (int)input.a;
 output[1] = (int)input.b;
}

kernel void test_kernel(test_struct input, global int* output) {
// CHECK: spir_kernel
// AMDGCN: define{{.*}} amdgpu_kernel void @test_kernel
// CHECK: struct.test_struct* nocapture {{.*}} byval(%struct.test_struct)
// CHECK: i32* nocapture %output
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
// CHECK: i64 %input.coerce0, i64 %input.coerce1, i32* nocapture %output
 output[0] = (int)input.a;
 output[1] = (int)input.b;
}
