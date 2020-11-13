// RUN: %clang %flags -shared -fPIC %s -o %T/first_tool.so
// RUN: %clang %flags -DTOOL -DSECOND_TOOL -shared -fPIC %s -o %T/second_tool.so
// RUN: %clang %flags -DTOOL -DTHIRD_TOOL -shared -fPIC %s -o %T/third_tool.so
// RUN: %libomp-compile -DCODE && env OMP_TOOL_LIBRARIES=%T/non_existing_file.so:%T/first_tool.so:%T/second_tool.so:%T/third_tool.so %libomp-run | FileCheck %s

// REQUIRES: ompt

/*
 *  This file contains code for three OMPT shared library tool to be 
 *  loaded and the code for the OpenMP executable. 
 *  No option enables code for the first shared library 
 *  (without an implementation of ompt_start_tool) during compilation
 *  -DTOOL -DSECOND_TOOL enables the code for the second tool during compilation
 *  -DTOOL -DTHIRD_TOOL enables the code for the third tool during compilation
 *  -DCODE enables the code for the executable during compilation
 */

#ifdef CODE
#include "stdio.h"
#include "omp.h"
#include "omp-tools.h"

int main()
{
  #pragma omp parallel num_threads(2)
  {
    #pragma omp master
    {
      int result = omp_control_tool(omp_control_tool_start, 0, NULL);
      printf("0: control_tool()=%d\n", result);
    }
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 
  
  // CHECK: {{^}}0: Do not initialize tool

  // CHECK: {{^}}0: Do initialize tool
  // CHECK: {{^}}0: Tool initialized
  // CHECK: {{^}}0: ompt_event_thread_begin
  // CHECK-DAG: {{^}}0: ompt_event_thread_begin
  // CHECK-DAG: {{^}}0: control_tool()=-1
  // CHECK: {{^}}0: Tool finalized
  

  return 0;
}

#endif /* CODE */

#ifdef TOOL

#include <omp-tools.h>
#include "stdio.h"

#ifdef SECOND_TOOL
// The second tool has an implementation of ompt_start_tool that returns NULL
ompt_start_tool_result_t* ompt_start_tool(
  unsigned int omp_version,
  const char *runtime_version)
{
  printf("0: Do not initialize tool\n");
  return NULL;
}
#elif defined(THIRD_TOOL)
// The third tool has an implementation of ompt_start_tool that returns a 
// pointer to a valid instance of ompt_start_tool_result_t

static void
on_ompt_callback_thread_begin(
  ompt_thread_t thread_type,
  ompt_data_t *thread_data)
{
  printf("0: ompt_event_thread_begin\n");
}

int ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                    ompt_data_t *tool_data) {
  ompt_set_callback_t ompt_set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");
  ompt_set_callback(ompt_callback_thread_begin, (ompt_callback_t)on_ompt_callback_thread_begin);
  printf("0: Tool initialized\n");
  return 1;
}

void ompt_finalize(ompt_data_t *tool_data)
{
  printf("0: Tool finalized\n");
}

ompt_start_tool_result_t* ompt_start_tool(
  unsigned int omp_version,
  const char *runtime_version)
{
  printf("0: Do initialize tool\n");
  static ompt_start_tool_result_t ompt_start_tool_result = {&ompt_initialize,&ompt_finalize, 0};
  return &ompt_start_tool_result;
}
#endif

#endif /* TOOL */
