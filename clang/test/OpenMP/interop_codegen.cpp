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
  omp_interop_t obj1 = omp_interop_none;
  omp_interop_t obj2 = omp_interop_none;
  omp_interop_t i1 = omp_interop_none;
  omp_interop_t i2 = omp_interop_none;
  omp_interop_t i3 = omp_interop_none;
  omp_interop_t i4 = omp_interop_none;
  omp_interop_t i5 = omp_interop_none;

  #pragma omp interop init(targetsync: obj1) init(targetsync: obj2)
  int id = (int )omp_get_interop_int(obj1, omp_ipr_fr_id, NULL);
  int id1 = (int )omp_get_interop_int(obj2, omp_ipr_fr_id, NULL);

  #pragma omp interop init(target,targetsync: i1) use(i2) use(i3) destroy(i4) destroy(i5)
  int id2 = (int )omp_get_interop_int(i1, omp_ipr_fr_id, NULL);
  int id3 = (int )omp_get_interop_int(i2, omp_ipr_fr_id, NULL);


}
#endif

// CHECK-LABEL: define {{.+}}main{{.+}}
// CHECK: call {{.+}}__tgt_interop_init({{.+}}obj1{{.*}})
// CHECK: call {{.+}}__tgt_interop_init({{.+}}obj2{{.*}})
// CHECK: call {{.+}}__tgt_interop_init({{.+}}i1{{.*}})
// CHECK: call {{.+}}__tgt_interop_destroy({{.+}}i4{{.*}})
// CHECK: call {{.+}}__tgt_interop_destroy({{.+}}i5{{.*}})
// CHECK: call {{.+}}__tgt_interop_use({{.+}}i2{{.*}})
// CHECK: call {{.+}}__tgt_interop_use({{.+}}i3{{.*}})
