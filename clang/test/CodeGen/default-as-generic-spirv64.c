// RUN: %clang_cc1 -triple spirv64 -fspirv-default-addr-space-is-generic -emit-llvm -o - %s | FileCheck %s --check-prefix=GENERIC

// Test that -fspirv-default-addr-space-is-generic causes unqualified pointers to be
// in the generic address space (addrspace(4)) for spirv64 targets.

void test_memcpy(void *dst, const void *src, unsigned long n) {
  __builtin_memcpy(dst, src, n);
}

// GENERIC: define spir_func void @test_memcpy(ptr addrspace(4) noundef %dst, ptr addrspace(4) noundef %src, i64 noundef %n)
// GENERIC: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 1 %{{.*}}, ptr addrspace(4) align 1 %{{.*}}, i64 %{{.*}}, i1 false)

void test_pointer(int *p) {
  *p = 42;
}

// GENERIC: define spir_func void @test_pointer(ptr addrspace(4) noundef %p)
