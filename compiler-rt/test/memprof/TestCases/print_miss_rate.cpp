// Check print_mem_info_cache_miss_rate and
// print_mem_info_cache_miss_rate_details options.

// RUN: %clangxx_memprof -O0 %s -o %t
// RUN: %env_memprof_opts=print_mem_info_cache_miss_rate=1 %run %t 2>&1 | FileCheck %s
// RUN: %env_memprof_opts=print_mem_info_cache_miss_rate=1:print_mem_info_cache_miss_rate_details=1 %run %t 2>&1 | FileCheck %s --check-prefix=DETAILS

// CHECK: Overall miss rate: 0 / {{.*}} = 0.00%
// DETAILS: Set 0 miss rate: 0 / {{.*}} = 0.00%
// DETAILS: Set 16380 miss rate: 0 / {{.*}} = 0.00%

int main() {
  return 0;
}
