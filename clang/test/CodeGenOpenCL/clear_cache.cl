// RUN: %clang_cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple spirv32-unknown-unknown -cl-std=CL2.0 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// The declaration of __builtin___clear_cache is rewritten to take the address
// space of each pointer it is called with, and in OpenCL that is almost never
// the target default address space, so @llvm.clear_cache has to be declared in
// the address space that is actually passed in.

// CHECK-LABEL: define spir_func void @clear_cache_global(
// CHECK:         call void @llvm.clear_cache.p1(ptr addrspace(1) %{{.*}}, ptr addrspace(1) %{{.*}})
// CHECK:       declare void @llvm.clear_cache.p1(ptr addrspace(1), ptr addrspace(1))
void clear_cache_global(global char *begin, global char *end) {
  __builtin___clear_cache(begin, end);
}

// CHECK-LABEL: define spir_func void @clear_cache_local(
// CHECK:         call void @llvm.clear_cache.p3(ptr addrspace(3) %{{.*}}, ptr addrspace(3) %{{.*}})
// CHECK:       declare void @llvm.clear_cache.p3(ptr addrspace(3), ptr addrspace(3))
void clear_cache_local(local char *begin, local char *end) {
  __builtin___clear_cache(begin, end);
}

// CHECK-LABEL: define spir_func void @clear_cache_generic(
// CHECK:         call void @llvm.clear_cache.p4(ptr addrspace(4) %{{.*}}, ptr addrspace(4) %{{.*}})
// CHECK:       declare void @llvm.clear_cache.p4(ptr addrspace(4), ptr addrspace(4))
void clear_cache_generic(generic char *begin, generic char *end) {
  __builtin___clear_cache(begin, end);
}

// The two pointers delimit one range, so a call that mixes address spaces gets
// the end pointer cast into the address space of the begin pointer.
// CHECK-LABEL: define spir_func void @clear_cache_mixed(
// CHECK:         [[CAST:%.*]] = addrspacecast ptr addrspace(3) %{{.*}} to ptr addrspace(1)
// CHECK:         call void @llvm.clear_cache.p1(ptr addrspace(1) %{{.*}}, ptr addrspace(1) [[CAST]])
void clear_cache_mixed(global char *begin, local char *end) {
  __builtin___clear_cache(begin, end);
}
