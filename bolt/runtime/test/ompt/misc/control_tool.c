// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
#define TEST_NEED_PRINT_FRAME_FROM_OUTLINED_FN
#include "callback.h"
#include <omp.h>

int main()
{
  #pragma omp parallel num_threads(1)
  {
    print_frame_from_outlined_fn(1);
    print_frame(0);
    omp_control_tool(omp_control_tool_flush, 1, NULL);
    print_current_address(0);
  }

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_control_tool'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: __builtin_frame_address({{.}})=[[EXIT_FRAME:0x[0-f]*]]
  // CHECK: {{^}}[[MASTER_ID]]: __builtin_frame_address(0)=[[REENTER_FRAME:0x[0-f]*]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_control_tool: command=3, modifier=1, arg=[[NULL]], codeptr_ra=[[RETURN_ADDRESS:0x[0-f]*]], current_task_frame.exit=[[EXIT_FRAME]], current_task_frame.reenter={{0x[0-f]*}}
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: current_address={{.*}}[[RETURN_ADDRESS]]

  return 0;
}
