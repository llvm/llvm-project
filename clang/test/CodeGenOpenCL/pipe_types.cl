// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O0 -cl-std=CL2.0 -o - %s | FileCheck %s

// CHECK: %opencl.pipe_t = type opaque
typedef unsigned char __attribute__((ext_vector_type(3))) uchar3;
typedef int __attribute__((ext_vector_type(4))) int4;

void test1(read_only pipe int p) {
// CHECK: define void @test1(%opencl.pipe_t* %p)
  reserve_id_t rid;
// CHECK: %rid = alloca %opencl.reserve_id_t
}

void test2(write_only pipe float p) {
// CHECK: define void @test2(%opencl.pipe_t* %p)
}

void test3(read_only pipe const int p) {
// CHECK: define void @test3(%opencl.pipe_t* %p)
}

void test4(read_only pipe uchar3 p) {
// CHECK: define void @test4(%opencl.pipe_t* %p)
}

void test5(read_only pipe int4 p) {
// CHECK: define void @test5(%opencl.pipe_t* %p)
}

typedef read_only pipe int MyPipe;
kernel void test6(MyPipe p) {
// CHECK: define spir_kernel void @test6(%opencl.pipe_t* %p)
}

struct Person {
  const char *Name;
  bool isFemale;
  int ID;
};

void test_reserved_read_pipe(global struct Person *SDst,
                             read_only pipe struct Person SPipe) {
// CHECK: define void @test_reserved_read_pipe
  read_pipe (SPipe, SDst);
  // CHECK: call i32 @__read_pipe_2(%opencl.pipe_t* %{{.*}}, i8* %{{.*}}, i32 16, i32 8)
  read_pipe (SPipe, SDst);
  // CHECK: call i32 @__read_pipe_2(%opencl.pipe_t* %{{.*}}, i8* %{{.*}}, i32 16, i32 8)
}
