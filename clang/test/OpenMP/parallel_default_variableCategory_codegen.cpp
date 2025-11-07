// RUN: %clangxx -DOMP60 -Xclang -verify -Wno-vla -fopenmp -fopenmp-version=60 -x c++ -S -emit-llvm %s -o - | FileCheck --check-prefixes=OMP60 %s
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

// OMP60-LABEL: define {{.*}}main.omp_outlined{{.*}}
// OMP60-NEXT:  entry:
// OMP60: %x.addr = alloca{{.*}}
// OMP60: %xPtr = alloca{{.*}}
// OMP60: store ptr null, ptr %xPtr{{.*}}
// OMP60: store ptr %xPtr{{.*}}
// OMP60: store ptr %x.addr{{.*}}
// OMP60-NEXT:  {{.*}}call{{.*}}__kmpc_omp_task_alloc{{.*}}
// OMP60: ret void
//
// OMP60: define {{.*}}main.omp_outlined{{.*}}
// OMP60-NEXT:  entry:
// OMP60: %i.addr = alloca{{.*}}
// OMP60-NEXT:  %n.addr = alloca{{.*}}
// OMP60-NEXT:  %aggregate.addr = alloca{{.*}}
// OMP60-NEXT:  %x.addr = alloca{{.*}}
// OMP60-NEXT:  %arr.addr = alloca{{.*}}
// OMP60: [[TMP0:%.*]] = load{{.*}}%i.addr{{.*}}
// OMP60-NEXT:  [[TMP1:%.*]] = load{{.*}}%n.addr{{.*}}
// OMP60-NEXT:  [[TMP2:%.*]] = load{{.*}}%aggregate.addr{{.*}}
// OMP60-NEXT:  [[TMP3:%.*]] = load{{.*}}%x.addr{{.*}}
// OMP60-NEXT:  [[TMP4:%.*]] = load{{.*}}%arr.addr{{.*}}
// OMP60: store ptr [[TMP2]]{{.*}}
// OMP60-NEXT:  {{.*}}call{{.*}}__kmpc_omp_task_alloc{{.*}}
// OMP60: store ptr [[TMP2]]{{.*}}
// OMP60: store ptr [[TMP3]]{{.*}}
// OMP60-NEXT:  {{.*}}call{{.*}}__kmpc_omp_task_alloc{{.*}}
// OMP60: store ptr [[TMP4]]{{.*}}
// OMP60-NEXT:  {{.*}}call{{.*}}__kmpc_omp_task_alloc{{.*}}
// OMP60: store ptr [[TMP4]]{{.*}}
// OMP60: store ptr [[TMP3]]{{.*}}
// OMP60-NEXT:  {{.*}}call{{.*}}__kmpc_omp_task_alloc{{.*}}
// OMP60: store ptr [[TMP0]]{{.*}}
// OMP60: store ptr [[TMP1]]{{.*}}
// OMP60: store ptr [[TMP2]]{{.*}}
// OMP60: store ptr [[TMP3]]{{.*}}
// OMP60-NEXT:  {{.*}}call{{.*}}__kmpc_omp_task_alloc{{.*}}
// OMP60: ret void
