// RUN: %clang_cc1 -O1 -triple spirv64 -fsycl-is-device %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -O1 -triple spirv64 -cl-std=CL3.0 -x cl %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -O1 -triple spirv32 -cl-std=CL3.0 -x cl %s -emit-llvm -o - | FileCheck %s

// CHECK: spir_func noundef ptr @test_cast_to_private(
// CHECK-SAME: ptr addrspace(4) noundef readnone [[P:%.*]]
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[SPV_CAST:%.*]] = tail call noundef ptr @llvm.spv.generic.cast.to.ptr.explicit.p0(ptr addrspace(4) %p)
// CHECK-NEXT:    ret ptr [[SPV_CAST]]
//
__attribute__((opencl_private)) int* test_cast_to_private(int* p) {
    return __builtin_spirv_generic_cast_to_ptr_explicit(p, 7);
}

// CHECK: spir_func noundef ptr addrspace(1) @test_cast_to_global(
// CHECK-SAME: ptr addrspace(4) noundef readnone [[P:%.*]]
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[SPV_CAST:%.*]] = tail call noundef ptr addrspace(1) @llvm.spv.generic.cast.to.ptr.explicit.p1(ptr addrspace(4) %p)
// CHECK-NEXT:    ret ptr addrspace(1) [[SPV_CAST]]
//
__attribute__((opencl_global)) int* test_cast_to_global(int* p) {
    return __builtin_spirv_generic_cast_to_ptr_explicit(p, 5);
}

// CHECK: spir_func noundef ptr addrspace(3) @test_cast_to_local(
// CHECK-SAME: ptr addrspace(4) noundef readnone [[P:%.*]]
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[SPV_CAST:%.*]] = tail call noundef ptr addrspace(3) @llvm.spv.generic.cast.to.ptr.explicit.p3(ptr addrspace(4) %p)
// CHECK-NEXT:    ret ptr addrspace(3) [[SPV_CAST]]
//
__attribute__((opencl_local)) int* test_cast_to_local(int* p) {
    return __builtin_spirv_generic_cast_to_ptr_explicit(p, 4);
}
