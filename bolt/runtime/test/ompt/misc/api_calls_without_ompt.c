// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt

#define _BSD_SOURCE
#define _DEFAULT_SOURCE

#include <stdio.h>
#include <inttypes.h>
#include <omp.h>
#include <omp-tools.h>

static ompt_set_callback_t ompt_set_callback;
static ompt_get_callback_t ompt_get_callback;
static ompt_get_state_t ompt_get_state;
static ompt_get_task_info_t ompt_get_task_info;
static ompt_get_thread_data_t ompt_get_thread_data;
static ompt_get_parallel_info_t ompt_get_parallel_info;
static ompt_get_unique_id_t ompt_get_unique_id;
static ompt_get_num_procs_t ompt_get_num_procs;
static ompt_get_num_places_t ompt_get_num_places;
static ompt_get_place_proc_ids_t ompt_get_place_proc_ids;
static ompt_get_place_num_t ompt_get_place_num;
static ompt_get_partition_place_nums_t ompt_get_partition_place_nums;
static ompt_get_proc_id_t ompt_get_proc_id;
static ompt_enumerate_states_t ompt_enumerate_states;
static ompt_enumerate_mutex_impls_t ompt_enumerate_mutex_impls;

int main() {
  // Call OpenMP API function to force initialization of OMPT.
  // (omp_get_thread_num() does not work because it just returns 0 if the
  // runtime isn't initialized yet...)
  omp_get_num_threads();

  ompt_data_t *tdata = ompt_get_thread_data();
  uint64_t tvalue = tdata ? tdata->value : 0;

  printf("%" PRIu64 ": ompt_get_num_places()=%d\n", tvalue,
         ompt_get_num_places());

  printf("%" PRIu64 ": ompt_get_place_proc_ids()=%d\n", tvalue,
         ompt_get_place_proc_ids(0, 0, NULL));

  printf("%" PRIu64 ": ompt_get_place_num()=%d\n", tvalue,
         ompt_get_place_num());

  printf("%" PRIu64 ": ompt_get_partition_place_nums()=%d\n", tvalue,
         ompt_get_partition_place_nums(0, NULL));

  printf("%" PRIu64 ": ompt_get_proc_id()=%d\n", tvalue, ompt_get_proc_id());

  printf("%" PRIu64 ": ompt_get_num_procs()=%d\n", tvalue,
         ompt_get_num_procs());

  ompt_callback_t callback;
  printf("%" PRIu64 ": ompt_get_callback()=%d\n", tvalue,
         ompt_get_callback(ompt_callback_thread_begin, &callback));

  printf("%" PRIu64 ": ompt_get_state()=%d\n", tvalue, ompt_get_state(NULL));

  int state = ompt_state_undefined;
  const char *state_name;
  printf("%" PRIu64 ": ompt_enumerate_states()=%d\n", tvalue,
         ompt_enumerate_states(state, &state, &state_name));

  int impl = ompt_mutex_impl_none;
  const char *impl_name;
  printf("%" PRIu64 ": ompt_enumerate_mutex_impls()=%d\n", tvalue,
         ompt_enumerate_mutex_impls(impl, &impl, &impl_name));

  printf("%" PRIu64 ": ompt_get_thread_data()=%p\n", tvalue,
         ompt_get_thread_data());

  printf("%" PRIu64 ": ompt_get_parallel_info()=%d\n", tvalue,
         ompt_get_parallel_info(0, NULL, NULL));

  printf("%" PRIu64 ": ompt_get_task_info()=%d\n", tvalue,
         ompt_get_task_info(0, NULL, NULL, NULL, NULL, NULL));

  // Check if libomp supports the callbacks for this test.

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_get_num_places()={{[0-9]+}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_place_proc_ids()={{[0-9]+}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_place_num()=-1

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_partition_place_nums()=0

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_proc_id()=-1

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_num_procs()={{[0-9]+}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_callback()=0

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_state()=0

  // CHECK: {{^}}[[MASTER_ID]]: ompt_enumerate_states()=1

  // CHECK: {{^}}[[MASTER_ID]]: ompt_enumerate_mutex_impls()=1

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_thread_data()=[[NULL]]

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_parallel_info()=0

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_task_info()=0

  return 0;
}

int ompt_initialize(ompt_function_lookup_t lookup, ompt_data_t *tool_data) {
  ompt_set_callback = (ompt_set_callback_t)lookup("ompt_set_callback");
  ompt_get_callback = (ompt_get_callback_t)lookup("ompt_get_callback");
  ompt_get_state = (ompt_get_state_t)lookup("ompt_get_state");
  ompt_get_task_info = (ompt_get_task_info_t)lookup("ompt_get_task_info");
  ompt_get_thread_data = (ompt_get_thread_data_t)lookup("ompt_get_thread_data");
  ompt_get_parallel_info =
      (ompt_get_parallel_info_t)lookup("ompt_get_parallel_info");
  ompt_get_unique_id = (ompt_get_unique_id_t)lookup("ompt_get_unique_id");

  ompt_get_num_procs = (ompt_get_num_procs_t)lookup("ompt_get_num_procs");
  ompt_get_num_places = (ompt_get_num_places_t)lookup("ompt_get_num_places");
  ompt_get_place_proc_ids =
      (ompt_get_place_proc_ids_t)lookup("ompt_get_place_proc_ids");
  ompt_get_place_num = (ompt_get_place_num_t)lookup("ompt_get_place_num");
  ompt_get_partition_place_nums =
      (ompt_get_partition_place_nums_t)lookup("ompt_get_partition_place_nums");
  ompt_get_proc_id = (ompt_get_proc_id_t)lookup("ompt_get_proc_id");
  ompt_enumerate_states =
      (ompt_enumerate_states_t)lookup("ompt_enumerate_states");
  ompt_enumerate_mutex_impls =
      (ompt_enumerate_mutex_impls_t)lookup("ompt_enumerate_mutex_impls");

  printf("0: NULL_POINTER=%p\n", (void *)NULL);
  return 0; // no success -> OMPT not enabled
}

void ompt_finalize(ompt_data_t *tool_data) {
  printf("0: ompt_event_runtime_shutdown\n");
}

ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
  static ompt_start_tool_result_t ompt_start_tool_result = {&ompt_initialize,
                                                            &ompt_finalize, 0};
  return &ompt_start_tool_result;
}
