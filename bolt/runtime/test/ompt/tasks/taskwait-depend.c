// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt

// taskwait with depend clause was introduced with gcc-9
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8

// clang does not yet support taskwait with depend clause
// clang-12 introduced parsing, but no codegen
// update expected result when codegen in clang was added
// XFAIL: clang

#include "callback.h"
#include <omp.h>

int main() {
  int x = 0;
#pragma omp parallel num_threads(2)
  {
#pragma omp master
    {
      print_ids(0);
      printf("%" PRIu64 ": address of x: %p\n", ompt_get_thread_data()->value,
             &x);
#pragma omp task depend(out : x)
      { x++; }
      print_fuzzy_address(1);
      #pragma omp taskwait depend(in: x)
      print_fuzzy_address(2);
    }
  }

  return 0;
}

// Check if libomp supports the callbacks for this test.
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_dependences'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_depende

// CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]

// make sure initial data pointers are null
// CHECK-NOT: 0: new_task_data initially not null

// CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_implicit_task_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID:[0-9]+]],
// CHECK-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]

// CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]],
// CHECK-SAME: task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT:0x[0-f]+]],
// CHECK-SAME: reenter_frame=[[NULL]]

// CHECK: {{^}}[[MASTER_ID]]: address of x: [[ADDRX:0x[0-f]+]]

// CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
// CHECK-SAME: parent_task_id={{[0-9]+}}, parent_task_frame.exit=[[EXIT]],
// CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}},
// CHECK-SAME: new_task_id=[[FIRST_TASK:[0-f]+]],
// CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}},
// CHECK-SAME: task_type=ompt_task_explicit=4, has_dependences=yes

// CHECK: {{^}}[[MASTER_ID]]: ompt_event_dependences:
// CHECK-SAME: task_id=[[FIRST_TASK]], deps=[([[ADDRX]],
// CHECK-SAME: ompt_dependence_type_inout)], ndeps=1

// CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]

// CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
// CHECK-SAME: parent_task_id={{[0-9]+}}, parent_task_frame.exit=[[EXIT]],
// CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}},
// CHECK-SAME: new_task_id=[[SECOND_TASK:[0-f]+]],
// CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}},
// CHECK-SAME: task_type=ompt_task_explicit|ompt_task_undeferred|
// CHECK-SAME: ompt_task_mergeable=1207959556, has_dependences=yes

// CHECK: {{^}}[[MASTER_ID]]: ompt_event_dependences:
// CHECK-SAME: task_id=[[SECOND_TASK]], deps=[([[ADDRX]],
// CHECK-SAME: ompt_dependence_type_in)], ndeps=1

// CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_end: task_id=[[SECOND_TASK]]

// CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]
