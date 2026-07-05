// expected-no-diagnostics
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

#define NNN 50
int aaa[NNN];

void parallel_taskgroup_loop() {
  #pragma omp parallel
  {
    #pragma omp taskgroup
    for (int i = 0 ; i < 2 ; i++) {
      #pragma omp loop
      for (int j = 0 ; j < NNN ; j++) {
        aaa[j] = j*NNN;
      }
    }
  }
}

void parallel_taskwait_loop() {
  #pragma omp parallel
  {
    #pragma omp taskwait
    for (int i = 0 ; i < 2 ; i++) {
      #pragma omp loop
      for (int j = 0 ; j < NNN ; j++) {
        aaa[j] = j*NNN;
      }
    }
  }
}

void parallel_single_loop() {
  #pragma omp parallel
  {
    for (int i = 0 ; i < 2 ; i++) {
      #pragma omp single
      #pragma omp loop
      for (int j = 0 ; j < NNN ; j++) {
        aaa[j] = j*NNN;
      }
    }
  }
}

void parallel_order_loop() {
  #pragma omp parallel
  {
    #pragma omp for order(concurrent)
    {
      for (int i = 0 ; i < 2 ; i++) {
        #pragma omp loop
        for (int j = 0 ; j < NNN ; j++) {
          aaa[j] = j*NNN;
        }
      }
    }
  }
}


void parallel_cancel_loop(bool flag) {
  #pragma omp ordered
  for (int i = 0 ; i < 2 ; i++) {
    #pragma omp parallel
    {
      #pragma omp cancel parallel if(flag)
      aaa[0] = 0;
      #pragma omp loop bind(parallel)
      for (int j = 0 ; j < NNN ; j++) {
        aaa[j] = j*NNN;
      }
    }
  }
}

int
main(int argc, char *argv[]) {
  parallel_taskgroup_loop();
  parallel_taskwait_loop();
  parallel_single_loop();
  parallel_order_loop();
  parallel_cancel_loop(true);
  parallel_cancel_loop(false);

  return 0;
}
// CHECK-LABEL: define {{.+}}parallel_taskgroup_loop{{.+}} {
// CHECK:        call {{.+}}__kmpc_fork_call({{.+}}parallel_taskgroup_loop{{.+}}.omp_outlined{{.*}}
//
//
// CHECK-LABEL: define {{.+}}parallel_taskgroup_loop{{.+}}.omp_outlined{{.+}} {
// CHECK:        call {{.+}}__kmpc_taskgroup
// CHECK:       for.body:
// CHECK:       omp.inner.for.cond:
// CHECK:       omp.inner.for.body:
// CHECK:       omp.inner.for.inc:
// CHECK:       omp.inner.for.end:
// CHECK:       for.end:
// CHECK:         call {{.+}}__kmpc_end_taskgroup
//
//
// CHECK-LABEL: define {{.+}}parallel_taskwait_loop{{.+}} {
// CHECK:         call {{.+}}__kmpc_fork_call({{.+}}parallel_taskwait_loop{{.+}}.omp_outlined{{.*}})
//
//
// CHECK-LABEL: define {{.+}}parallel_taskwait_loop{{.+}}.omp_outlined{{.+}} {
// CHECK:         [[TMP2:%.*]] = {{.+}}__kmpc_omp_taskwait
// CHECK:       for.cond:
// CHECK:       for.body:
// CHECK:         {{.+}}__kmpc_for_static_init_4
// CHECK:       omp.inner.for.cond:
// CHECK:       omp.inner.for.body:
// CHECK:       omp.body.continue:
// CHECK: 	omp.inner.for.inc:
// CHECK: 	omp.inner.for.end:
// CHECK: 	omp.loop.exit:
// CHECK:         call void @__kmpc_for_static_fini
// CHECK:         call void @__kmpc_barrier
// CHECK:       for.end:
//
//
// CHECK-LABEL: define {{.+}}parallel_single_loop{{.+}} {
// CHECK:         call {{.+}}__kmpc_fork_call({{.+}}parallel_single_loop{{.+}}.omp_outlined{{.*}})
//
//
// CHECK-LABEL: define {{.+}}parallel_single_loop{{.+}}.omp_outlined{{.+}} {
// CHECK:       for.body:
// CHECK:         [[TMP3:%.*]] = call {{.+}}__kmpc_single
// CHECK:       omp.inner.for.end:
// CHECK:         call {{.+}}__kmpc_end_single
// CHECK:       omp_if.end:
// CHECK:        call {{.+}}__kmpc_barrier
// CHECK:       for.end:
//
//
// CHECK-LABEL: define {{.+}}parallel_order_loop{{.+}} {
// CHECK:         call {{.+}}__kmpc_fork_call({{.+}}parallel_order_loop{{.+}}.omp_outlined{{.*}})
//
//
// CHECK-LABEL: define {{.+}}parallel_order_loop{{.+}}.omp_outlined{{.+}} {
// CHECK:        call {{.+}}__kmpc_for_static_init_4
// CHECK:       omp.inner.for.body:
// CHECK:       omp.loop.exit:
// CHECK:        call {{.+}}__kmpc_for_static_fini
// CHECK:        call {{.+}}__kmpc_barrier
//
//
// CHECK-LABEL: define {{.+}}parallel_cancel_loop{{.+}} {
// CHECK:         call {{.+}}__kmpc_ordered
// CHECK:       for.body:
// CHECK:         call {{.+}}__kmpc_fork_call({{.+}}parallel_cancel_loop{{.+}}.omp_outlined{{.+}})
// CHECK:       for.end:
// CHECK:         call {{.+}}__kmpc_end_ordered
//
//
// CHECK-LABEL: define {{.+}}parallel_cancel_loop{{.+}}.omp_outlined{{.+}} {
// CHECK:       omp_if.then:
// CHECK:         [[TMP4:%.*]] = call {{.+}}__kmpc_cancel
// CHECK:       .cancel.exit:
// CHECK:         [[TMP8:%.*]] = call {{.+}}__kmpc_cancel_barrier
// CHECK:       omp_if.end:
// CHECK:         call {{.+}}__kmpc_for_static_init_4
// CHECK:       omp.inner.for.body:
// CHECK:       omp.loop.exit:
// CHECK:         call {{.+}}__kmpc_for_static_fini
// CHECK:         [[TMP24:%.*]] = call {{.+}}__kmpc_cancel_barrier
// CHECK:       .cancel.continue5:
//
//
// CHECK-LABEL: define {{.+}}@main{{.+}} {
// CHECK:         call {{.+}}parallel_taskgroup_loop{{.+}}()
// CHECK-NEXT:    call {{.+}}parallel_taskwait_loop{{.+}}()
// CHECK-NEXT:    call {{.+}}parallel_single_loop{{.+}}()
// CHECK-NEXT:    call {{.+}}parallel_order_loop{{.+}}()
// CHECK-NEXT:    call {{.+}}parallel_cancel_loop{{.+}}({{.+}}true)
// CHECK-NEXT:    call {{.+}}parallel_cancel_loop{{.+}}({{.+}}false)
//
