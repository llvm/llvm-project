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
// CHECK-LABEL: define dso_local void @{{.+}}parallel_loop
// CHECK-NEXT:  entry:
// CHECK-NEXT:    call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @[[GLOB3:[0-9]+]], i32 0, ptr {{.+}}parallel_loop{{.+}}.omp_outlined{{.*}})
// CHECK-NEXT:    ret void
//
//
// CHECK-LABEL: define internal void {{.+}}parallel_loop{{.+}}.omp_outlined
// CHECK-SAME: (ptr noalias noundef [[DOTGLOBAL_TID_:%.*]], ptr noalias noundef [[DOTBOUND_TID_:%.*]]) #[[ATTR1:[0-9]+]] {
// CHECK:         call void @__kmpc_for_static_init_4
// CHECK:       omp.inner.for.body:
// CHECK:       omp.loop.exit:
// CHECK-NEXT:    call void @__kmpc_for_static_fini
// CHECK-NEXT:    call void @__kmpc_barrier
// CHECK-NEXT:    ret void
//
//
// CHECK-LABEL: define dso_local void {{.+}}parallel_loop_orphan{{.+}}
// CHECK-NEXT:  entry:
// CHECK:         [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num
// CHECK:         call void @__kmpc_for_static_init_4
// CHECK:       omp.inner.for.body:
// CHECK:       omp.inner.for.end:
// CHECK:       omp.loop.exit:
// CHECK-NEXT:    call void @__kmpc_for_static_fini
// CHECK-NEXT:    call void @__kmpc_barrier
// CHECK-NEXT:    ret void
//
//
// CHECK-LABEL: define dso_local void {{.+}}teams_loop{{.+}}
// CHECK-NEXT:  entry:
// CHECK-NEXT:    call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr @[[GLOB3]], i32 0, ptr {{.+}}teams_loop{{.+}}.omp_outlined{{.*}})
// CHECK-NEXT:    ret void
//
//
// CHECK-LABEL: define internal void {{.+}}teams_loop{{.+}}.omp_outlined{{.+}}
// CHECK-NEXT:  entry:
// CHECK:         call void @__kmpc_for_static_init_4
// CHECK:       omp.inner.for.body:
// CHECK:       omp.loop.exit:
// CHECK-NEXT:    call void @__kmpc_for_static_fini
//
//
// CHECK-LABEL: define dso_local void {{.+}}thread_loop{{.+}}
// CHECK-NEXT:  entry:
// CHECK-NEXT:    call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @[[GLOB3]], i32 0, ptr {{.+}}thread_loop{{.+}}.omp_outlined{{.*}})
// CHECK-NEXT:    ret void
//
//
// CHECK-LABEL: define internal void {{.+}}thread_loop{{.+}}.omp_outlined{{.+}}
// CHECK-NEXT:  entry:
// CHECK:       omp.inner.for.body:
// CHECK:       omp.inner.for.end:
//
//
// CHECK-LABEL: define dso_local void {{.+}}thread_loop_orphan{{.+}}
// CHECK-NEXT:  entry:
// CHECK:       omp.inner.for.cond:
// CHECK:       omp.inner.for.body:
// CHECK:       omp.inner.for.end:
//
//
// CHECK-LABEL: define {{.+}}main{{.+}}
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca i32, align 4
// CHECK:         call void {{.+}}parallel_loop{{.+}}()
// CHECK-NEXT:    call void {{.+}}parallel_loop_orphan{{.+}}()
// CHECK-NEXT:    call void {{.+}}teams_loop{{.+}}()
// CHECK-NEXT:    call void {{.+}}thread_loop{{.+}}()
// CHECK-NEXT:    call void {{.+}}thread_loop_orphan{{.+}}()
// CHECK-NEXT:    ret i32 0
//
