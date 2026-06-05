// RUN: %clang_cc1 -triple spirv64-intel -emit-llvm -o - %s | FileCheck %s

// Test we can call the intrinsics with non-zero AS pointers.

typedef int v8i __attribute__((ext_vector_type(8)));
typedef _Bool v8b __attribute__((ext_vector_type(8)));

// CHECK-LABEL: define spir_func <8 x i32> @test_load_expand(
// CHECK: call addrspace(9) <8 x i32> @llvm.masked.expandload.v8i32.p4(ptr addrspace(4) %{{.*}}, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
v8i test_load_expand(v8b m, int *p, v8i t) {
  return __builtin_masked_expand_load(m, p, t);
}

// CHECK-LABEL: define spir_func void @test_compress_store(
// CHECK: call addrspace(9) void @llvm.masked.compressstore.v8i32.p4(<8 x i32> %{{.*}}, ptr addrspace(4) %{{.*}}, <8 x i1> %{{.*}})
void test_compress_store(v8b m, v8i v, int *p) {
  __builtin_masked_compress_store(m, v, p);
}
