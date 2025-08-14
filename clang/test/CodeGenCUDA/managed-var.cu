// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=COMMON,DEV,HIP-D,HIP-NORDC-D %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -std=c++11 \
// RUN:   -emit-llvm -fgpu-rdc -cuid=abc -o - -x hip %s > %t.dev
// RUN: cat %t.dev | FileCheck -check-prefixes=COMMON,DEV,HIP-D,HIP-RDC-D %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=COMMON,HOST,HIP-H,NORDC %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -std=c++11 \
// RUN:   -emit-llvm -fgpu-rdc -cuid=abc -o - -x hip %s > %t.host
// RUN: cat %t.host | FileCheck -check-prefixes=COMMON,HOST,HIP-H,RDC,HIP-RDC %s

// Check device and host compilation use the same postfix for static
// variable name.

// RUN: cat %t.dev %t.host | FileCheck -check-prefix=POSTFIX %s

// RUN: %clang_cc1 -triple nvptx64 -fcuda-is-device -std=c++11 \
// RUN:   -emit-llvm -o - -x cuda %s | FileCheck \
// RUN:   -check-prefixes=COMMON,DEV,CUDA-D,CUDA-NORDC-D %s

// RUN: %clang_cc1 -triple nvptx64 -fcuda-is-device -std=c++11 \
// RUN:   -emit-llvm -fgpu-rdc -cuid=abc -o - -x cuda %s > %t.dev
// RUN: cat %t.dev | FileCheck -check-prefixes=COMMON,DEV,CUDA-D,CUDA-RDC-D %s

// RUN: echo "GPU binary" > %t.fatbin
// RUN: %clang_cc1 -triple x86_64-gnu-linux -std=c++11 \
// RUN:   -emit-llvm -o - -x cuda %s -fcuda-include-gpubinary %t.fatbin \
// RUN:   | FileCheck -check-prefixes=COMMON,HOST,CUDA-H,NORDC,CUDA-NORDC %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -std=c++11 \
// RUN:   -emit-llvm -fgpu-rdc -cuid=abc -o - -x cuda %s \
// RUN:   -fcuda-include-gpubinary %t.fatbin > %t.host
// RUN: cat %t.host \
// RUN:   | FileCheck -check-prefixes=COMMON,HOST,CUDA-H,RDC,CUDA-RDC %s

// Check device and host compilation use the same postfix for static
// variable name.

// RUN: cat %t.dev %t.host | FileCheck -check-prefix=CUDA-POSTFIX %s

#include "Inputs/cuda.h"

struct vec {
  float x,y,z;
};

// HIP-D-DAG: @x.managed = addrspace(1) externally_initialized global i32 1, align 4
// HIP-D-DAG: @x = addrspace(1) externally_initialized global ptr addrspace(1) null
// CUDA-D-DAG: @x = addrspace(1) externally_initialized global i32 1, align 4
// NORDC-DAG: @x.managed = internal global i32 1
// RDC-DAG: @x.managed = global i32 1
// NORDC-DAG: @x = internal externally_initialized global ptr null
// RDC-DAG: @x = externally_initialized global ptr null
// HOST-DAG: @[[DEVNAMEX:[0-9]+]] = {{.*}}c"x\00"
__managed__ int x = 1;

// HIP-D-DAG: @v.managed = addrspace(1) externally_initialized global [100 x %struct.vec] zeroinitializer, align 4
// HIP-D-DAG: @v = addrspace(1) externally_initialized global ptr addrspace(1) null
// CUDA-D-DAG: @v = addrspace(1) externally_initialized global [100 x %struct.vec] zeroinitializer, align 4
__managed__ vec v[100];

// HIP-D-DAG: @v2.managed = addrspace(1) externally_initialized global <{ %struct.vec, [99 x %struct.vec] }> <{ %struct.vec { float 1.000000e+00, float 1.000000e+00, float 1.000000e+00 }, [99 x %struct.vec] zeroinitializer }>, align 4
// HIP-D-DAG: @v2 = addrspace(1) externally_initialized global ptr addrspace(1) null
// CUDA-D-DAG: @v2 = addrspace(1) externally_initialized global <{ %struct.vec, [99 x %struct.vec] }> <{ %struct.vec { float 1.000000e+00, float 1.000000e+00, float 1.000000e+00 }, [99 x %struct.vec] zeroinitializer }>, align 4
__managed__ vec v2[100] = {{1, 1, 1}};

// HIP-D-DAG: @ex.managed = external addrspace(1) global i32, align 4
// HIP-D-DAG: @ex = external addrspace(1) externally_initialized global ptr addrspace(1)
// CUDA-D-DAG: @ex = external addrspace(1) global i32, align 4
// HOST-DAG: @ex.managed = external global i32
// HOST-DAG: @ex = external externally_initialized global ptr
extern __managed__ int ex;

// HIP-NORDC-D-DAG: @_ZL2sx.managed = addrspace(1) externally_initialized global i32 1, align 4
// HIP-NORDC-D-DAG: @_ZL2sx = addrspace(1) externally_initialized global ptr addrspace(1) null
// CUDA-NORDC-D-DAG: @_ZL2sx = addrspace(1) externally_initialized global i32 1, align 4
// HIP-RDC-D-DAG: @_ZL2sx.static.[[HASH:.*]].managed = addrspace(1) externally_initialized global i32 1, align 4
// HIP-RDC-D-DAG: @_ZL2sx.static.[[HASH]] = addrspace(1) externally_initialized global ptr addrspace(1) null
// CUDA-RDC-D-DAG: @_ZL2sx__static__[[HASH:.*]] = addrspace(1) externally_initialized global i32 1, align 4
// 
// HOST-DAG: @_ZL2sx.managed = internal global i32 1
// HOST-DAG: @_ZL2sx = internal externally_initialized global ptr null
// NORDC-DAG: @[[DEVNAMESX:[0-9]+]] = {{.*}}c"_ZL2sx\00"
// HIP-RDC-DAG: @[[DEVNAMESX:[0-9]+]] = {{.*}}c"_ZL2sx.static.[[HASH:.*]]\00"
// CUDA-RDC-DAG: @[[DEVNAMESX:[0-9]+]] = {{.*}}c"_ZL2sx__static__[[HASH:.*]]\00"

// POSTFIX:  @_ZL2sx.static.[[HASH:.*]] = addrspace(1) externally_initialized global ptr addrspace(1) null
// POSTFIX: @[[DEVNAMESX:[0-9]+]] = {{.*}}c"_ZL2sx.static.[[HASH]]\00"

// CUDA-POSTFIX:  @_ZL2sx__static__[[HASH:.*]] = addrspace(1) externally_initialized global i32 1, align 4
// CUDA-POSTFIX: @[[DEVNAMESX:[0-9]+]] = {{.*}}c"_ZL2sx__static__[[HASH]]\00"
static __managed__ int sx = 1;

// DEV-DAG: @llvm.compiler.used
// DEV-SAME-DAG: @x.managed
// DEV-SAME-DAG: @x
// DEV-SAME-DAG: @v.managed
// DEV-SAME-DAG: @v
// DEV-SAME-DAG: @_ZL2sx.managed
// DEV-SAME-DAG: @_ZL2sx

// Force ex and sx mitted in device compilation.
__global__ void foo(int *z) {
  *z = x + ex + sx;
  v[1].x = 2;
}

// Force ex and sx emitted in host compilatioin.
int foo2() {
  return ex + sx;
}

// COMMON-LABEL: define {{.*}}@_Z4loadv()
// HIP-D:  %ld.managed = load ptr addrspace(1), ptr addrspace(1) @x, align 4
// HIP-D:  %0 = addrspacecast ptr addrspace(1) %ld.managed to ptr
// HIP-D:  [[RET:%.*]] = load i32, ptr %0, align 4
// CUDA-D: [[RET:%.*]] = load i32, ptr addrspacecast (ptr addrspace(1) @x to ptr), align 4
// DEV:  ret i32 [[RET]]
// HOST:  %ld.managed = load ptr, ptr @x, align 4
// HOST:  %0 = load i32, ptr %ld.managed, align 4
// HOST:  ret i32 %0
__device__ __host__ int load() {
  return x;
}

// COMMON-LABEL: define {{.*}}@_Z5storev()
// HIP-D:  %ld.managed = load ptr addrspace(1), ptr addrspace(1) @x, align 4
// HIP-D:  %0 = addrspacecast ptr addrspace(1) %ld.managed to ptr
// HIP-D:  store i32 2, ptr %0, align 4
// CUDA-D: store i32 2, ptr addrspacecast (ptr addrspace(1) @x to ptr), align 4
// HOST:  %ld.managed = load ptr, ptr @x, align 4
// HOST:  store i32 2, ptr %ld.managed, align 4
__device__ __host__ void store() {
  x = 2;
}

// COMMON-LABEL: define {{.*}}@_Z10addr_takenv()
// HIP-D:  %0 = addrspacecast ptr addrspace(1) %ld.managed to ptr
// HIP-D:  store ptr %0, ptr %p.ascast, align 8
// CUDA-D: store ptr addrspacecast (ptr addrspace(1) @x to ptr), ptr %p, align 8
// DEV:  [[LOAD:%.*]] = load ptr, ptr {{%p.*}}, align 8
// DEV:  store i32 3, ptr [[LOAD]], align 4
// HOST:  %ld.managed = load ptr, ptr @x, align 4
// HOST:  store ptr %ld.managed, ptr %p, align 8
// HOST:  %0 = load ptr, ptr %p, align 8
// HOST:  store i32 3, ptr %0, align 4
__device__ __host__ void addr_taken() {
  int *p = &x;
  *p = 3;
}

// HOST-LABEL: define {{.*}}@_Z5load2v()
// HOST: %ld.managed = load ptr, ptr @v, align 16
// HOST:  %0 = getelementptr inbounds [100 x %struct.vec], ptr %ld.managed, i64 0, i64 1
// HOST:  %1 = load float, ptr %0, align 4
// HOST:  ret float %1
__device__ __host__ float load2() {
  return v[1].x;
}

// HOST-LABEL: define {{.*}}@_Z5load3v()
// HOST:  %ld.managed = load ptr, ptr @v2, align 16
// HOST:  %0 = getelementptr inbounds [100 x %struct.vec], ptr %ld.managed, i64 0, i64 1
// HOST:  %1 = getelementptr inbounds nuw %struct.vec, ptr %0, i32 0, i32 1
// HOST:  %2 = load float, ptr %1, align 4
// HOST:  ret float %2
float load3() {
  return v2[1].y;
}

// HOST-LABEL: define {{.*}}@_Z11addr_taken2v()
// HOST:  %ld.managed = load ptr, ptr @v, align 16
// HOST:  %0 = getelementptr inbounds [100 x %struct.vec], ptr %ld.managed, i64 0, i64 1
// HOST:  %1 = ptrtoint ptr %0 to i64
// HOST:  %ld.managed1 = load ptr, ptr @v2, align 16
// HOST:  %2 = getelementptr inbounds [100 x %struct.vec], ptr %ld.managed1, i64 0, i64 1
// HOST:  %3 = getelementptr inbounds nuw %struct.vec, ptr %2, i32 0, i32 1
// HOST:  %4 = ptrtoint ptr %3 to i64
// HOST:  %5 = sub i64 %4, %1
// HOST:  %sub.ptr.div = sdiv exact i64 %5, 4
// HOST:  %conv = sitofp i64 %sub.ptr.div to float
// HOST:  ret float %conv
float addr_taken2() {
  return (float)reinterpret_cast<long>(&(v2[1].y)-&(v[1].x));
}

// COMMON-LABEL: define {{.*}}@_Z5load4v()
// HIP-D:  %ld.managed = load ptr addrspace(1), ptr addrspace(1) @ex, align 4
// HIP-D:  %0 = addrspacecast ptr addrspace(1) %ld.managed to ptr
// HIP-D:  [[LOAD:%.*]] = load i32, ptr %0, align 4
// CUDA-D:  [[LOAD:%.*]] = load i32, ptr addrspacecast (ptr addrspace(1) @ex to ptr), align 4
// DEV:  ret i32 [[LOAD]]
// HOST:  %ld.managed = load ptr, ptr @ex, align 4
// HOST:  %0 = load i32, ptr %ld.managed, align 4
// HOST:  ret i32 %0
__device__ __host__ int load4() {
  return ex;
}

// HIP-H-DAG: __hipRegisterManagedVar({{.*}}, ptr @x, ptr @x.managed, ptr @[[DEVNAMEX]], i64 4, i32 4)
// HIP-H-DAG: __hipRegisterManagedVar({{.*}}, ptr @_ZL2sx, ptr @_ZL2sx.managed, ptr @[[DEVNAMESX]]
// HIP-H-NOT: __hipRegisterManagedVar({{.*}}, ptr @ex, ptr @ex.managed
// HIP-H-DAG: declare void @__hipRegisterManagedVar(ptr, ptr, ptr, ptr, i64, i32)

// CUDA-H-DAG: __cudaRegisterManagedVar({{.*}}, ptr @x, ptr @[[DEVNAMEX]], ptr @[[DEVNAMEX]], i32 0, i32 4, i32 0, i32 0)
// CUDA-H-DAG: __cudaRegisterManagedVar({{.*}}, ptr @_ZL2sx,  ptr @[[DEVNAMESX]], ptr @[[DEVNAMESX]]
// CUDA-H-NOT: __cudaRegisterManagedVar({{.*}}, ptr @ex
// CUDA-H-DAG: declare void @__cudaRegisterManagedVar(ptr, ptr, ptr, ptr, i32, i32, i32, i32)
// CUDA-NORDC-DAG: __cudaInitModule
