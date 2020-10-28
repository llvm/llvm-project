// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
#define USE_PRIVATE_TOOL 1
#include "callback.h"
#include <omp.h>

int main() {
  int x;
#pragma omp parallel num_threads(2)
  {
#pragma omp master
    {
#pragma omp task
      { x++; }
#pragma omp task firstprivate(x)
      { x++; }
    }
  }

  return 0;
}

static void on_ompt_callback_implicit_task(ompt_scope_endpoint_t endpoint,
                                           ompt_data_t *parallel_data,
                                           ompt_data_t *task_data,
                                           unsigned int team_size,
                                           unsigned int thread_num, int flag) {
  void *addr = NULL;
  size_t size = 0;
  int result = ompt_get_task_memory(&addr, &size, 0);
  switch (endpoint) {
  case ompt_scope_begin:
    task_data->value = ompt_get_unique_id();
    printf("ompt_event_implicit_task_begin: task_id=%" PRIu64
           ", memory_addr=%p, memory_size=%lu, result=%d \n",
           task_data->value, addr, size, result);
    break;
  case ompt_scope_end:
    printf("ompt_event_implicit_task_end: task_id=%" PRIu64
           ", memory_addr=%p, memory_size=%lu, result=%d \n",
           task_data->value, addr, size, result);
    break;
  }
}

static void
on_ompt_callback_task_create(ompt_data_t *encountering_task_data,
                             const ompt_frame_t *encountering_task_frame,
                             ompt_data_t *new_task_data, int flags,
                             int has_dependences, const void *codeptr_ra) {
  if (flags & ompt_task_initial)
    return; // not interested in the initial task
  new_task_data->value = ompt_get_unique_id();
  void *addr = NULL;
  size_t size = 0;
  printf("ompt_event_task_create: task_id=%" PRIu64 "\n", new_task_data->value);
}

static void on_ompt_callback_task_schedule(ompt_data_t *first_task_data,
                                           ompt_task_status_t prior_task_status,
                                           ompt_data_t *second_task_data) {
  void *addr = NULL;
  size_t size = 0;
  int result = ompt_get_task_memory(&addr, &size, 0);
  printf("ompt_event_task_schedule: task_id=%" PRIu64
         ", memory_addr=%p, memory_size=%lu, result=%d\n",
         first_task_data->value, addr, size, result);
}

int ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                    ompt_data_t *tool_data) {
  ompt_set_callback = (ompt_set_callback_t)lookup("ompt_set_callback");
  ompt_get_unique_id = (ompt_get_unique_id_t)lookup("ompt_get_unique_id");
  ompt_get_task_memory = (ompt_get_task_memory_t)lookup("ompt_get_task_memory");

  register_callback(ompt_callback_implicit_task);
  register_callback(ompt_callback_task_create);
  register_callback(ompt_callback_task_schedule);
  printf("0: NULL_POINTER=%p\n", (void *)NULL);
  return 1; // success
}

void ompt_finalize(ompt_data_t *tool_data) {}

ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
  static ompt_start_tool_result_t ompt_start_tool_result = {&ompt_initialize,
                                                            &ompt_finalize, 0};
  return &ompt_start_tool_result;
}

// CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]

// CHECK: ompt_event_implicit_task_begin: task_id=[[TASK_ID:[0-9]+]]
// CHECK-SAME: memory_addr=[[NULL]], memory_size=0, result=0

// CHECK: ompt_event_task_create: task_id=[[TASK_ID_0:[0-9]+]]
// CHECK-DAG: ompt_event_task_create: task_id=[[TASK_ID_1:[0-9]+]]

// Expects non-zero address, size, and result
// CHECK-DAG: ompt_event_task_schedule: task_id=[[TASK_ID_0]],
// memory_addr=0x{{[0-f]+}}, memory_size={{[1-9][0-9]*}}, result=1
// CHECK-DAG: ompt_event_task_schedule: task_id=[[TASK_ID_1]],
// memory_addr=0x{{[0-f]+}}, memory_size={{[1-9][0-9]*}}, result=1

// CHECK: ompt_event_implicit_task_end: task_id=[[TASK_ID]]
// CHECK-SAME: memory_addr=[[NULL]], memory_size=0, result=0
