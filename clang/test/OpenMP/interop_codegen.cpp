// expected-no-diagnostics
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1  -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s

#ifndef HEADER
#define HEADER

typedef void *omp_interop_t;
#define omp_interop_none 0
#define omp_ipr_fr_id -1
typedef long omp_intptr_t;
#define NULL 0

extern omp_intptr_t omp_get_interop_int(const omp_interop_t, int, int *);

int main() {
  omp_interop_t obj = omp_interop_none;
  omp_interop_t i1 = omp_interop_none;
  omp_interop_t i2 = omp_interop_none;
  omp_interop_t i3 = omp_interop_none;
  omp_interop_t i4 = omp_interop_none;
  omp_interop_t i5 = omp_interop_none;

  #pragma omp interop init(targetsync: i1) init(targetsync: obj)
  int id = (int )omp_get_interop_int(obj, omp_ipr_fr_id, NULL);
  int id1 = (int )omp_get_interop_int(i1, omp_ipr_fr_id, NULL);


}
#endif

// CHECK-LABEL: define {{.+}}main{{.+}} 
// CHECK: call {{.+}}__tgt_interop_init({{.+}}i1{{.*}})
// CHECK: call {{.+}}__tgt_interop_init({{.+}}obj{{.*}})
