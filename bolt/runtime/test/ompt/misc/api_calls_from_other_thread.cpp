// RUN: %libomp-cxx-compile-and-run | FileCheck %s
// REQUIRES: ompt, linux

#include <thread>
#include "callback.h"

void f() {
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
}

int main() {
#pragma omp parallel num_threads(1)
  {}

  std::thread t1(f);
  t1.join();

  // Check if libomp supports the callbacks for this test.

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_get_num_places()={{[0-9]+}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_place_proc_ids()={{[0-9]+}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_place_num()=-1

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_partition_place_nums()=0

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_proc_id()=-1

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_num_procs()={{[0-9]+}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_callback()=1

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_state()=0

  // CHECK: {{^}}[[MASTER_ID]]: ompt_enumerate_states()=1

  // CHECK: {{^}}[[MASTER_ID]]: ompt_enumerate_mutex_impls()=1

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_thread_data()=[[NULL]]

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_parallel_info()=0

  // CHECK: {{^}}[[MASTER_ID]]: ompt_get_task_info()=0

  return 0;
}
