// RUN: %clangxx -Xclang -verify -Wno-vla -fopenmp -fopenmp-version=60 -x c++ -S -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#include <vector>

int global;
#define VECTOR_SIZE 4
int main (int argc, char **argv) {
  int i,n;
  int x;

  n = VECTOR_SIZE;

  #pragma omp parallel masked firstprivate(x) num_threads(2)
  {
     int *xPtr = nullptr;
     // scalar
     #pragma omp task default(shared:scalar)
     {
       xPtr = &x;
     }
     #pragma omp taskwait

     // pointer
     #pragma omp task default(shared:pointer) shared(x)
     {
       xPtr = &x;
     }
     #pragma omp taskwait
  }

  int *aggregate[VECTOR_SIZE] = {0,0,0,0};
  std::vector<int *> arr(VECTOR_SIZE,0);
  
  #pragma omp parallel masked num_threads(2)
  {
     // aggregate
     #pragma omp task default(shared:aggregate)
     for(i=0;i<n;i++) {
       aggregate[i] = &x;
     }
     #pragma omp taskwait

     #pragma omp task default(shared:aggregate) shared(x)
     for(i=0;i<n;i++) {
       aggregate[i] = &x;
     }
     #pragma omp taskwait

     // allocatable
     #pragma omp task default(shared:allocatable)
     for(i=0;i<n;i++) {
       arr[i] = &x;
     }
     #pragma omp taskwait

     #pragma omp task default(shared:allocatable) shared(x)
     for(i=0;i<n;i++) {
       arr[i] = &x;
     }
     #pragma omp taskwait

     // all
     #pragma omp task default(shared:all)
     for(i=0;i<n;i++) {
       aggregate[i] = &x;
     }
     #pragma omp taskwait
  }
}

#endif

// CHECK-LABEL: define {{.*}}main.omp_outlined{{.*}}
// CHECK-NEXT:  entry:
// CHECK: %x.addr = alloca{{.*}}
// CHECK: %xPtr = alloca{{.*}}
// CHECK: store ptr null, ptr %xPtr{{.*}}
// CHECK: store ptr %xPtr{{.*}}
// CHECK: store ptr %x.addr{{.*}}
// CHECK-NEXT:  {{.*}}call{{.*}}__kmpc_omp_task_alloc{{.*}}
// CHECK: ret void
//
// CHECK: define {{.*}}main.omp_outlined{{.*}}
// CHECK-NEXT:  entry:
// CHECK-DAG: %i.addr = alloca{{.*}}
// CHECK-DAG:  %n.addr = alloca{{.*}}
// CHECK-DAG:  %aggregate.addr = alloca{{.*}}
// CHECK-DAG:  %x.addr = alloca{{.*}}
// CHECK-DAG:  %arr.addr = alloca{{.*}}
// CHECK: [[TMP0:%.*]] = load{{.*}}%i.addr{{.*}}
// CHECK-NEXT:  [[TMP1:%.*]] = load{{.*}}%n.addr{{.*}}
// CHECK-NEXT:  [[TMP2:%.*]] = load{{.*}}%aggregate.addr{{.*}}
// CHECK-NEXT:  [[TMP3:%.*]] = load{{.*}}%x.addr{{.*}}
// CHECK-NEXT:  [[TMP4:%.*]] = load{{.*}}%arr.addr{{.*}}
// CHECK: store ptr [[TMP2]]{{.*}}
// CHECK-NEXT:  {{.*}}call{{.*}}__kmpc_omp_task_alloc{{.*}}
// CHECK: store ptr [[TMP2]]{{.*}}
// CHECK: store ptr [[TMP3]]{{.*}}
// CHECK-NEXT:  {{.*}}call{{.*}}__kmpc_omp_task_alloc{{.*}}
// CHECK: store ptr [[TMP4]]{{.*}}
// CHECK-NEXT:  {{.*}}call{{.*}}__kmpc_omp_task_alloc{{.*}}
// CHECK: store ptr [[TMP4]]{{.*}}
// CHECK: store ptr [[TMP3]]{{.*}}
// CHECK-NEXT:  {{.*}}call{{.*}}__kmpc_omp_task_alloc{{.*}}
// CHECK: store ptr [[TMP0]]{{.*}}
// CHECK: store ptr [[TMP1]]{{.*}}
// CHECK: store ptr [[TMP2]]{{.*}}
// CHECK: store ptr [[TMP3]]{{.*}}
// CHECK-NEXT:  {{.*}}call{{.*}}__kmpc_omp_task_alloc{{.*}}
// CHECK: ret void
