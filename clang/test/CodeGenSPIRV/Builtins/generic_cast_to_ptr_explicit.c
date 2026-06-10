// RUN: %clang_cc1 -O1 -triple spirv64 -fsycl-is-device -x c++ %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -O1 -triple spirv64 -cl-std=CL3.0 -x cl %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -O1 -triple spirv32 -cl-std=CL3.0 -x cl %s -emit-llvm -o - | FileCheck %s

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_EXTERNAL [[clang::sycl_external]]
#define GLOBALAS [[clang::sycl_global]]
#define LOCALAS [[clang::sycl_local]]
#define PRIVATEAS [[clang::sycl_private]]
#else
#define SYCL_EXTERNAL
#define GLOBALAS __attribute__((opencl_global))
#define LOCALAS __attribute__((opencl_local))
#define PRIVATEAS __attribute__((opencl_private))
#endif

// CHECK: spir_func noundef ptr @{{.*}}test_cast_to_private{{.*}}(ptr addrspace(4) noundef readnone [[P:%.*]]
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[SPV_CAST:%.*]] = tail call noundef ptr @llvm.spv.generic.cast.to.ptr.explicit.p0(ptr addrspace(4) %p)
// CHECK-NEXT:    ret ptr [[SPV_CAST]]
//
SYCL_EXTERNAL int PRIVATEAS * test_cast_to_private(int* p) {
    return __builtin_spirv_generic_cast_to_ptr_explicit(p, 7);
}

// CHECK: spir_func noundef ptr addrspace(1) @{{.*}}test_cast_to_global{{.*}}(ptr addrspace(4) noundef readnone [[P:%.*]]
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[SPV_CAST:%.*]] = tail call noundef ptr addrspace(1) @llvm.spv.generic.cast.to.ptr.explicit.p1(ptr addrspace(4) %p)
// CHECK-NEXT:    ret ptr addrspace(1) [[SPV_CAST]]
//
SYCL_EXTERNAL int GLOBALAS * test_cast_to_global(int* p) {
    return __builtin_spirv_generic_cast_to_ptr_explicit(p, 5);
}

// CHECK: spir_func noundef ptr addrspace(3) @{{.*}}test_cast_to_local{{.*}}(ptr addrspace(4) noundef readnone [[P:%.*]]
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[SPV_CAST:%.*]] = tail call noundef ptr addrspace(3) @llvm.spv.generic.cast.to.ptr.explicit.p3(ptr addrspace(4) %p)
// CHECK-NEXT:    ret ptr addrspace(3) [[SPV_CAST]]
//
SYCL_EXTERNAL int LOCALAS * test_cast_to_local(int* p) {
    return __builtin_spirv_generic_cast_to_ptr_explicit(p, 4);
}
