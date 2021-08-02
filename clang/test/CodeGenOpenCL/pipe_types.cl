// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=CL2.0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=CL3.0 -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,+__opencl_c_program_scope_global_variables -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=CL3.0 -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,-__opencl_c_program_scope_global_variables -o - %s | FileCheck %s

// CHECK: %opencl.pipe_ro_t = type opaque
// CHECK: %opencl.pipe_wo_t = type opaque
typedef unsigned char __attribute__((ext_vector_type(3))) uchar3;
typedef int __attribute__((ext_vector_type(4))) int4;

void test1(read_only pipe int p) {
// CHECK: define{{.*}} void @test1(%opencl.pipe_ro_t* %p)
  reserve_id_t rid;
// CHECK: %rid = alloca %opencl.reserve_id_t
}

void test2(write_only pipe float p) {
// CHECK: define{{.*}} void @test2(%opencl.pipe_wo_t* %p)
}

void test3(read_only pipe const int p) {
// CHECK: define{{.*}} void @test3(%opencl.pipe_ro_t* %p)
}

void test4(read_only pipe uchar3 p) {
// CHECK: define{{.*}} void @test4(%opencl.pipe_ro_t* %p)
}

void test5(read_only pipe int4 p) {
// CHECK: define{{.*}} void @test5(%opencl.pipe_ro_t* %p)
}

typedef read_only pipe int MyPipe;
kernel void test6(MyPipe p) {
// CHECK: define{{.*}} spir_kernel void @test6(%opencl.pipe_ro_t* %p)
}
