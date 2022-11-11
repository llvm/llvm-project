// RUN: %clang_cc1 -no-opaque-pointers -triple amdgcn-amd-amdhsa -fcuda-is-device -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=COMMON,DEV,NORDC-D %s

// RUN: %clang_cc1 -no-opaque-pointers -triple amdgcn-amd-amdhsa -fcuda-is-device -std=c++11 \
// RUN:   -emit-llvm -fgpu-rdc -cuid=abc -o - -x hip %s > %t.dev
// RUN: cat %t.dev | FileCheck -check-prefixes=COMMON,DEV,RDC-D %s

// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-gnu-linux -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=COMMON,HOST,NORDC %s

// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-gnu-linux -std=c++11 \
// RUN:   -emit-llvm -fgpu-rdc -cuid=abc -o - -x hip %s > %t.host
// RUN: cat %t.host | FileCheck -check-prefixes=COMMON,HOST,RDC %s

// Check device and host compilation use the same postfix for static
// variable name.

// RUN: cat %t.dev %t.host | FileCheck -check-prefix=POSTFIX %s

#include "Inputs/cuda.h"

struct vec {
  float x,y,z;
};

// DEV-DAG: @x.managed = addrspace(1) externally_initialized global i32 1, align 4
// DEV-DAG: @x = addrspace(1) externally_initialized global i32 addrspace(1)* null
// NORDC-DAG: @x.managed = internal global i32 1
// RDC-DAG: @x.managed = global i32 1
// NORDC-DAG: @x = internal externally_initialized global i32* null
// RDC-DAG: @x = externally_initialized global i32* null
// HOST-DAG: @[[DEVNAMEX:[0-9]+]] = {{.*}}c"x\00"
__managed__ int x = 1;

// DEV-DAG: @v.managed = addrspace(1) externally_initialized global [100 x %struct.vec] zeroinitializer, align 4
// DEV-DAG: @v = addrspace(1) externally_initialized global [100 x %struct.vec] addrspace(1)* null
__managed__ vec v[100];

// DEV-DAG: @v2.managed = addrspace(1) externally_initialized global <{ %struct.vec, [99 x %struct.vec] }> <{ %struct.vec { float 1.000000e+00, float 1.000000e+00, float 1.000000e+00 }, [99 x %struct.vec] zeroinitializer }>, align 4
// DEV-DAG: @v2 = addrspace(1) externally_initialized global <{ %struct.vec, [99 x %struct.vec] }> addrspace(1)* null
__managed__ vec v2[100] = {{1, 1, 1}};

// DEV-DAG: @ex.managed = external addrspace(1) global i32, align 4
// DEV-DAG: @ex = external addrspace(1) externally_initialized global i32 addrspace(1)*
// HOST-DAG: @ex.managed = external global i32
// HOST-DAG: @ex = external externally_initialized global i32*
extern __managed__ int ex;

// NORDC-D-DAG: @_ZL2sx.managed = addrspace(1) externally_initialized global i32 1, align 4
// NORDC-D-DAG: @_ZL2sx = addrspace(1) externally_initialized global i32 addrspace(1)* null
// RDC-D-DAG: @_ZL2sx.static.[[HASH:.*]].managed = addrspace(1) externally_initialized global i32 1, align 4
// RDC-D-DAG: @_ZL2sx.static.[[HASH]] = addrspace(1) externally_initialized global i32 addrspace(1)* null
// HOST-DAG: @_ZL2sx.managed = internal global i32 1
// HOST-DAG: @_ZL2sx = internal externally_initialized global i32* null
// NORDC-DAG: @[[DEVNAMESX:[0-9]+]] = {{.*}}c"_ZL2sx\00"
// RDC-DAG: @[[DEVNAMESX:[0-9]+]] = {{.*}}c"_ZL2sx.static.[[HASH:.*]]\00"

// POSTFIX:  @_ZL2sx.static.[[HASH:.*]] = addrspace(1) externally_initialized global i32 addrspace(1)* null
// POSTFIX: @[[DEVNAMESX:[0-9]+]] = {{.*}}c"_ZL2sx.static.[[HASH]]\00"
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
// DEV:  %ld.managed = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* @x, align 4
// DEV:  %0 = addrspacecast i32 addrspace(1)* %ld.managed to i32*
// DEV:  %1 = load i32, i32* %0, align 4
// DEV:  ret i32 %1
// HOST:  %ld.managed = load i32*, i32** @x, align 4
// HOST:  %0 = load i32, i32* %ld.managed, align 4
// HOST:  ret i32 %0
__device__ __host__ int load() {
  return x;
}

// COMMON-LABEL: define {{.*}}@_Z5storev()
// DEV:  %ld.managed = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* @x, align 4
// DEV:  %0 = addrspacecast i32 addrspace(1)* %ld.managed to i32*
// DEV:  store i32 2, i32* %0, align 4
// HOST:  %ld.managed = load i32*, i32** @x, align 4
// HOST:  store i32 2, i32* %ld.managed, align 4
__device__ __host__ void store() {
  x = 2;
}

// COMMON-LABEL: define {{.*}}@_Z10addr_takenv()
// DEV:  %0 = addrspacecast i32 addrspace(1)* %ld.managed to i32*
// DEV:  store i32* %0, i32** %p.ascast, align 8
// DEV:  %1 = load i32*, i32** %p.ascast, align 8
// DEV:  store i32 3, i32* %1, align 4
// HOST:  %ld.managed = load i32*, i32** @x, align 4
// HOST:  store i32* %ld.managed, i32** %p, align 8
// HOST:  %0 = load i32*, i32** %p, align 8
// HOST:  store i32 3, i32* %0, align 4
__device__ __host__ void addr_taken() {
  int *p = &x;
  *p = 3;
}

// HOST-LABEL: define {{.*}}@_Z5load2v()
// HOST: %ld.managed = load [100 x %struct.vec]*, [100 x %struct.vec]** @v, align 16
// HOST:  %0 = getelementptr inbounds [100 x %struct.vec], [100 x %struct.vec]* %ld.managed, i64 0, i64 1, i32 0
// HOST:  %1 = load float, float* %0, align 4
// HOST:  ret float %1
__device__ __host__ float load2() {
  return v[1].x;
}

// HOST-LABEL: define {{.*}}@_Z5load3v()
// HOST:  %ld.managed = load <{ %struct.vec, [99 x %struct.vec] }>*, <{ %struct.vec, [99 x %struct.vec] }>** @v2, align 16
// HOST:  %0 = bitcast <{ %struct.vec, [99 x %struct.vec] }>* %ld.managed to [100 x %struct.vec]*
// HOST:  %1 = getelementptr inbounds [100 x %struct.vec], [100 x %struct.vec]* %0, i64 0, i64 1, i32 1
// HOST:  %2 = load float, float* %1, align 4
// HOST:  ret float %2
float load3() {
  return v2[1].y;
}

// HOST-LABEL: define {{.*}}@_Z11addr_taken2v()
// HOST:  %ld.managed = load [100 x %struct.vec]*, [100 x %struct.vec]** @v, align 16
// HOST:  %0 = getelementptr inbounds [100 x %struct.vec], [100 x %struct.vec]* %ld.managed, i64 0, i64 1, i32 0
// HOST:  %1 = ptrtoint float* %0 to i64
// HOST:  %ld.managed1 = load <{ %struct.vec, [99 x %struct.vec] }>*, <{ %struct.vec, [99 x %struct.vec] }>** @v2, align 16
// HOST:  %2 = bitcast <{ %struct.vec, [99 x %struct.vec] }>* %ld.managed1 to [100 x %struct.vec]*
// HOST:  %3 = getelementptr inbounds [100 x %struct.vec], [100 x %struct.vec]* %2, i64 0, i64 1, i32 1
// HOST:  %4 = ptrtoint float* %3 to i64
// HOST:  %5 = sub i64 %4, %1
// HOST:  %sub.ptr.div = sdiv exact i64 %5, 4
// HOST:  %conv = sitofp i64 %sub.ptr.div to float
// HOST:  ret float %conv
float addr_taken2() {
  return (float)reinterpret_cast<long>(&(v2[1].y)-&(v[1].x));
}

// COMMON-LABEL: define {{.*}}@_Z5load4v()
// DEV:  %ld.managed = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* @ex, align 4
// DEV:  %0 = addrspacecast i32 addrspace(1)* %ld.managed to i32*
// DEV:  %1 = load i32, i32* %0, align 4
// DEV:  ret i32 %1
// HOST:  %ld.managed = load i32*, i32** @ex, align 4
// HOST:  %0 = load i32, i32* %ld.managed, align 4
// HOST:  ret i32 %0
__device__ __host__ int load4() {
  return ex;
}

// HOST-DAG: __hipRegisterManagedVar({{.*}}@x {{.*}}@x.managed {{.*}}@[[DEVNAMEX]]{{.*}}, i64 4, i32 4)
// HOST-DAG: __hipRegisterManagedVar({{.*}}@_ZL2sx {{.*}}@_ZL2sx.managed {{.*}}@[[DEVNAMESX]]
// HOST-NOT: __hipRegisterManagedVar({{.*}}@ex {{.*}}@ex.managed
// HOST-DAG: declare void @__hipRegisterManagedVar(i8**, i8*, i8*, i8*, i64, i32)
