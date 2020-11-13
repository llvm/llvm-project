// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
#include "callback.h"
#include <omp.h>

int main()
{
  #pragma omp ordered
  {
    print_current_address(1);
    print_ids(0);
  }
  print_current_address(2);
  

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquire'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquired'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_released'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_nest_lock'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_wait_ordered: wait_id=[[WAIT_ID:[0-9]+]], hint={{[0-9]+}}, impl={{[0-9]+}}, codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_ordered: wait_id=[[WAIT_ID]], codeptr_ra=[[RETURN_ADDRESS]]
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: current_address={{.*}}[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_ordered: wait_id=[[WAIT_ID]], codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: current_address={{.*}}[[RETURN_ADDRESS]]
  
  return 0;
}
