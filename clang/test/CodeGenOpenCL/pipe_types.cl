// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=CL2.0 -o - %s | FileCheck --check-prefixes=CHECK,CHECK-STRUCT %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=CL3.0 -cl-ext=-all,+__opencl_c_pipes,+__opencl_c_generic_address_space,+__opencl_c_program_scope_global_variables -o - %s | FileCheck --check-prefixes=CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=CL3.0 -cl-ext=-all,+__opencl_c_pipes,+__opencl_c_generic_address_space -o - %s | FileCheck --check-prefixes=CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=clc++2021 -cl-ext=-all,+__opencl_c_pipes,+__opencl_c_generic_address_space,+__opencl_c_program_scope_global_variables -o - %s | FileCheck --check-prefixes=CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=clc++2021 -cl-ext=-all,+__opencl_c_pipes,+__opencl_c_generic_address_space -o - %s | FileCheck --check-prefixes=CHECK %s

typedef unsigned char __attribute__((ext_vector_type(3))) uchar3;
typedef int __attribute__((ext_vector_type(4))) int4;

void test1(read_only pipe int p) {
// CHECK: define{{.*}} void @{{.*}}test1{{.*}}(ptr %p)
  reserve_id_t rid;
// CHECK: %rid = alloca ptr
}

void test2(write_only pipe float p) {
// CHECK: define{{.*}} void @{{.*}}test2{{.*}}(ptr %p)
}

void test3(read_only pipe const int p) {
// CHECK: define{{.*}} void @{{.*}}test3{{.*}}(ptr %p)
}

void test4(read_only pipe uchar3 p) {
// CHECK: define{{.*}} void @{{.*}}test4{{.*}}(ptr %p)
}

void test5(read_only pipe int4 p) {
// CHECK: define{{.*}} void @{{.*}}test5{{.*}}(ptr %p)
}

typedef read_only pipe int MyPipe;
kernel void test6(MyPipe p) {
// CHECK: define{{.*}} spir_kernel void @test6(ptr %p)
}

struct Person {
  const char *Name;
  bool isFemale;
  int ID;
};

void test_reserved_read_pipe(global struct Person *SDst,
                             read_only pipe struct Person SPipe) {
  // CHECK-STRUCT: define{{.*}} void @test_reserved_read_pipe
  read_pipe (SPipe, SDst);
  // CHECK-STRUCT: call i32 @__read_pipe_2(ptr %{{.*}}, ptr %{{.*}}, i32 16, i32 8)
  read_pipe (SPipe, SDst);
  // CHECK-STRUCT: call i32 @__read_pipe_2(ptr %{{.*}}, ptr %{{.*}}, i32 16, i32 8)
}
