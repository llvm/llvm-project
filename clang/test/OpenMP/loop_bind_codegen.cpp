// expected-no-diagnostics
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s


#define NNN 50
int aaa[NNN];

void parallel_loop() {
  #pragma omp parallel
  {
    #pragma omp loop bind(parallel)
    for (int j = 0 ; j < NNN ; j++) {
      aaa[j] = j*NNN;
    }
  }
}

void parallel_loop_orphan() {
  #pragma omp loop bind(parallel)
  for (int j = 0 ; j < NNN ; j++) {
     aaa[j] = j*NNN;
  }
}


void teams_loop() {
  #pragma omp teams
  {
     #pragma omp loop bind(teams)
     for (int j = 0 ; j < NNN ; j++) {
       aaa[j] = j*NNN;
     }
   }
}

void thread_loop() {
  #pragma omp parallel
  {
     #pragma omp loop bind(thread)
     for (int j = 0 ; j < NNN ; j++) {
       aaa[j] = j*NNN;
     }
   }
}

void thread_loop_orphan() {
  #pragma omp loop bind(thread)
  for (int j = 0 ; j < NNN ; j++) {
    aaa[j] = j*NNN;
  }
}

int main() {
  parallel_loop();
  parallel_loop_orphan();
  teams_loop();
  thread_loop();
  thread_loop_orphan();

  return 0;
}
// CHECK-LABEL: define {{.+}}parallel_loop{{.+}}
// CHECK:    call {{.+}}__kmpc_fork_call({{.+}}parallel_loop{{.+}}.omp_outlined{{.*}})
//
//
// CHECK-LABEL: define {{.+}}parallel_loop{{.+}}.omp_outlined{{.+}}
// CHECK:         call {{.+}}__kmpc_for_static_init_4
// CHECK:       omp.inner.for.body:
// CHECK:       omp.loop.exit:
// CHECK-NEXT:    call {{.+}}__kmpc_for_static_fini
// CHECK-NEXT:    call {{.+}}__kmpc_barrier
//
//
// CHECK-LABEL: define {{.+}}parallel_loop_orphan{{.+}}
// CHECK:         [[TMP0:%.*]] = call {{.+}}__kmpc_global_thread_num
// CHECK:         call {{.+}}__kmpc_for_static_init_4
// CHECK:       omp.inner.for.body:
// CHECK:       omp.inner.for.end:
// CHECK:       omp.loop.exit:
// CHECK-NEXT:    call {{.+}}__kmpc_for_static_fini
// CHECK-NEXT:    call {{.+}}__kmpc_barrier
//
//
// CHECK-LABEL: define {{.+}}teams_loop{{.+}}
// CHECK:    call {{.+}}__kmpc_fork_teams({{.+}}teams_loop{{.+}}.omp_outlined{{.*}})
//
//
// CHECK-LABEL: define {{.+}}teams_loop{{.+}}.omp_outlined{{.+}}
// CHECK:         call {{.+}}__kmpc_for_static_init_4
// CHECK:       omp.inner.for.body:
// CHECK:       omp.loop.exit:
// CHECK-NEXT:    call {{.+}}__kmpc_for_static_fini
//
//
// CHECK-LABEL: define {{.+}}thread_loop{{.+}}
// CHECK:    call {{.+}}__kmpc_fork_call({{.+}}thread_loop{{.+}}.omp_outlined{{.*}})
//
//
// CHECK-LABEL: define {{.+}}thread_loop{{.+}}.omp_outlined{{.+}}
// CHECK:       omp.inner.for.body:
// CHECK:       omp.inner.for.end:
//
//
// CHECK-LABEL: define {{.+}}thread_loop_orphan{{.+}}
// CHECK:       omp.inner.for.cond:
// CHECK:       omp.inner.for.body:
// CHECK:       omp.inner.for.end:
//
//
// CHECK-LABEL: define {{.+}}main{{.+}}
// CHECK:         call {{.+}}parallel_loop{{.+}}()
// CHECK-NEXT:    call {{.+}}parallel_loop_orphan{{.+}}()
// CHECK-NEXT:    call {{.+}}teams_loop{{.+}}()
// CHECK-NEXT:    call {{.+}}thread_loop{{.+}}()
// CHECK-NEXT:    call {{.+}}thread_loop_orphan{{.+}}()
//
