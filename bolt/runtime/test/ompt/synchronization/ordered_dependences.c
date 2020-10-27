// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
#include "callback.h"
#include <omp.h>

int main() {
  int a[10][10];
  int i, j;
#pragma omp parallel num_threads(2)
#pragma omp for ordered(2)
  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++) {
      a[i][j] = i + j + 1;
      printf("%d, %d\n", i, j);
#pragma omp ordered depend(sink : i - 1, j) depend(sink : i, j - 1)
      if (i > 0 && j > 0)
        a[i][j] = a[i - 1][j] + a[i][j - 1] + 1;
      printf("%d, %d\n", i, j);
#pragma omp ordered depend(source)
    }

  return 0;
}
// CHECK: 0: NULL_POINTER=[[NULL:.*$]]

// CHECK: {{^}}[[MASTER:[0-9]+]]: ompt_event_loop_begin:
// CHECK-SAME: parallel_id={{[0-9]+}}, parent_task_id=[[ITASK:[0-9]+]],

// CHECK: {{^}}[[MASTER]]: ompt_event_dependences: task_id=[[ITASK]],
// CHECK-SAME: deps=[(0, ompt_dependence_type_source), (0,
// CHECK-SAME: ompt_dependence_type_source)], ndeps=2

// CHECK: {{^}}[[MASTER]]: ompt_event_dependences: task_id=[[ITASK]],
// CHECK-SAME: deps=[(0, ompt_dependence_type_sink), (0,
// CHECK-SAME: ompt_dependence_type_sink)], ndeps=2

// CHECK: {{^}}[[MASTER]]: ompt_event_dependences: task_id=[[ITASK]],
// CHECK-SAME: deps=[(0, ompt_dependence_type_source), (1,
// CHECK-SAME: ompt_dependence_type_source)], ndeps=2

// CHECK: {{^}}[[WORKER:[0-9]+]]: ompt_event_loop_begin:
// CHECK-SAME: parallel_id={{[0-9]+}}, parent_task_id=[[ITASK:[0-9]+]],

// CHECK: {{^}}[[WORKER]]: ompt_event_dependences: task_id=[[ITASK]],
// CHECK-SAME: deps=[(0, ompt_dependence_type_sink), (0,
// CHECK-SAME: ompt_dependence_type_sink)], ndeps=2

// CHECK: {{^}}[[WORKER]]: ompt_event_dependences: task_id=[[ITASK]],
// CHECK-SAME: deps=[(1, ompt_dependence_type_source), (0,
// CHECK-SAME: ompt_dependence_type_source)], ndeps=2

// either can be first for last iteration

// CHECK-DAG: [[ITASK]]{{.*}}deps=[(0{{.*}}sink), (1,{{.*}}sink)]

// CHECK-DAG: [[ITASK]]{{.*}}deps=[(1{{.*}}sink), (0,{{.*}}sink)]

// CHECK: {{^}}[[WORKER]]: ompt_event_dependences: task_id=[[ITASK]],
// CHECK-SAME: deps=[(1, ompt_dependence_type_source), (1,
// CHECK-SAME: ompt_dependence_type_source)], ndeps=2
