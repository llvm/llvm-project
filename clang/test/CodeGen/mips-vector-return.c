// RUN: %clang_cc1 -triple mipsel-unknown-linux -O3 -S -o - -emit-llvm %s | FileCheck %s -check-prefix=O32
// RUN: %clang_cc1 -triple mips64el-unknown-linux -O3 -S -target-abi n64 -o - -emit-llvm %s | FileCheck %s -check-prefix=N64

// vectors larger than 16-bytes are returned via the hidden pointer argument. 
// N64/N32 returns vectors whose size is equal to or smaller than 16-bytes in
// integer registers. 
typedef float  v4sf __attribute__ ((__vector_size__ (16)));
typedef double v4df __attribute__ ((__vector_size__ (32)));
typedef int v4i32 __attribute__ ((__vector_size__ (16)));

// O32-LABEL: define{{.*}} void @test_v4sf(<4 x float>* noalias nocapture writeonly sret
// N64: define{{.*}} inreg { i64, i64 } @test_v4sf
v4sf test_v4sf(float a) {
  return (v4sf){0.0f, a, 0.0f, 0.0f};
}

// O32-LABEL: define{{.*}} void @test_v4df(<4 x double>* noalias nocapture writeonly sret
// N64-LABEL: define{{.*}} void @test_v4df(<4 x double>* noalias nocapture writeonly sret
v4df test_v4df(double a) {
  return (v4df){0.0, a, 0.0, 0.0};
}

// O32 returns integer vectors whose size is equal to or smaller than 16-bytes
// in integer registers.
//
// O32: define{{.*}} inreg { i32, i32, i32, i32 } @test_v4i32
// N64: define{{.*}} inreg { i64, i64 } @test_v4i32
v4i32 test_v4i32(int a) {
  return (v4i32){0, a, 0, 0};
}

