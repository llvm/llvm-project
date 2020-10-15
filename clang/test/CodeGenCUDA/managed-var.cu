// REQUIRES: x86-registered-target, amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=DEV %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -std=c++11 \
// RUN:   -emit-llvm -fgpu-rdc -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=DEV %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=HOST %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -std=c++11 \
// RUN:   -emit-llvm -fgpu-rdc -o - -x hip %s | FileCheck \
// RUN:   -check-prefixes=HOST %s

#include "Inputs/cuda.h"

// DEV-DAG: @x = {{.*}}addrspace(1) externally_initialized global i32 undef
// HOST-DAG: @x = internal global i32 1
// HOST-DAG: @x.managed = internal global i32* null
// HOST-DAG: @[[DEVNAMEX:[0-9]+]] = {{.*}}c"x\00"

struct vec {
  float x,y,z;
};

__managed__ int x = 1;
__managed__ vec v[100];
__managed__ vec v2[100] = {{1, 1, 1}};

__global__ void foo(int *z) {
  *z = x;
  v[1].x = 2;
}

// HOST-LABEL: define {{.*}}@_Z4loadv()
// HOST:  %ld.managed = load i32*, i32** @x.managed, align 4
// HOST:  %0 = load i32, i32* %ld.managed, align 4
// HOST:  ret i32 %0
int load() {
  return x;
}

// HOST-LABEL: define {{.*}}@_Z5storev()
// HOST:  %ld.managed = load i32*, i32** @x.managed, align 4
// HOST:  store i32 2, i32* %ld.managed, align 4
void store() {
  x = 2;
}

// HOST-LABEL: define {{.*}}@_Z10addr_takenv()
// HOST:  %ld.managed = load i32*, i32** @x.managed, align 4
// HOST:  store i32* %ld.managed, i32** %p, align 8
// HOST:  %0 = load i32*, i32** %p, align 8
// HOST:  store i32 3, i32* %0, align 4
void addr_taken() {
  int *p = &x;
  *p = 3;
}

// HOST-LABEL: define {{.*}}@_Z5load2v()
// HOST: %ld.managed = load [100 x %struct.vec]*, [100 x %struct.vec]** @v.managed, align 16
// HOST:  %0 = getelementptr inbounds [100 x %struct.vec], [100 x %struct.vec]* %ld.managed, i64 0, i64 1, i32 0
// HOST:  %1 = load float, float* %0, align 4
// HOST:  ret float %1
float load2() {
  return v[1].x;
}

// HOST-LABEL: define {{.*}}@_Z5load3v()
// HOST:  %ld.managed = load <{ %struct.vec, [99 x %struct.vec] }>*, <{ %struct.vec, [99 x %struct.vec] }>** @v2.managed, align 16
// HOST:  %0 = bitcast <{ %struct.vec, [99 x %struct.vec] }>* %ld.managed to [100 x %struct.vec]*
// HOST:  %1 = getelementptr inbounds [100 x %struct.vec], [100 x %struct.vec]* %0, i64 0, i64 1, i32 1
// HOST:  %2 = load float, float* %1, align 4
// HOST:  ret float %2
float load3() {
  return v2[1].y;
}

// HOST-LABEL: define {{.*}}@_Z11addr_taken2v()
// HOST:  %ld.managed = load [100 x %struct.vec]*, [100 x %struct.vec]** @v.managed, align 16
// HOST:  %0 = getelementptr inbounds [100 x %struct.vec], [100 x %struct.vec]* %ld.managed, i64 0, i64 1, i32 0
// HOST:  %1 = ptrtoint float* %0 to i64
// HOST:  %ld.managed1 = load <{ %struct.vec, [99 x %struct.vec] }>*, <{ %struct.vec, [99 x %struct.vec] }>** @v2.managed, align 16
// HOST:  %2 = bitcast <{ %struct.vec, [99 x %struct.vec] }>* %ld.managed1 to [100 x %struct.vec]*
// HOST:  %3 = getelementptr inbounds [100 x %struct.vec], [100 x %struct.vec]* %2, i64 0, i64 1, i32 1
// HOST:  %4 = ptrtoint float* %3 to i64
// HOST:  %5 = sub i64 %4, %1
// HOST:  %6 = sdiv i64 %5, 4
// HOST:  %7 = sitofp i64 %6 to float
// HOST:  ret float %7
float addr_taken2() {
  return (float)reinterpret_cast<long>(&(v2[1].y)-&(v[1].x));
}

// HOST-DAG: __hipRegisterManagedVar({{.*}}@x.managed {{.*}}@x {{.*}}@[[DEVNAMEX]]{{.*}}, i64 4, i32 4)
// HOST-DAG: declare void @__hipRegisterManagedVar(i8**, i8*, i8*, i8*, i64, i32)
