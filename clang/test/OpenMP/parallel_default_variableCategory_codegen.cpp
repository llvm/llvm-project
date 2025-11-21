// RUN: %clangxx -Xclang -verify -Wno-vla -fopenmp -fopenmp-version=60 -x c++ -S -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

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
// CHECK: store ptr null, ptr{{.*}}
// CHECK-NEXT: {{.*}}getelementptr {{.*}}
// CHECK-NEXT: store ptr {{.*}}
// CHECK-NEXT: {{.*}}getelementptr {{.*}}
// CHECK-NEXT: store ptr {{.*}}
// CHECK-NEXT: {{.*}}call{{.*}}__kmpc_omp_task_alloc{{.*}}
// CHECK: ret void
//
// CHECK: define {{.*}}main.omp_outlined{{.*}}
// CHECK: {{.*}}getelementptr {{.*}}
// CHECK-NEXT: store ptr {{.*}}
// CHECK-NEXT: {{.*}}call{{.*}}__kmpc_omp_task_alloc{{.*}}
// CHECK: store ptr {{.*}}
// CHECK-NEXT: {{.*}}getelementptr {{.*}}
// CHECK-NEXT: store ptr {{.*}}
// CHECK-NEXT: {{.*}}call{{.*}}__kmpc_omp_task_alloc{{.*}}
// CHECK: store ptr {{.*}}
// CHECK-NEXT: {{.*}}getelementptr {{.*}}
// CHECK-NEXT: store ptr {{.*}}
// CHECK-NEXT: {{.*}}getelementptr {{.*}}
// CHECK-NEXT: store ptr {{.*}}
// CHECK-NEXT: {{.*}}getelementptr {{.*}}
// CHECK-NEXT: store ptr {{.*}}
// CHECK-NEXT: {{.*}}call{{.*}}__kmpc_omp_task_alloc{{.*}}
// CHECK: ret void
