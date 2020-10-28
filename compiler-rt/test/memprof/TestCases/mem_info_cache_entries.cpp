// Check mem_info_cache_entries option.

// RUN: %clangxx_memprof -O0 %s -o %t && %env_memprof_opts=mem_info_cache_entries=15:print_mem_info_cache_miss_rate=1:print_mem_info_cache_miss_rate_details=1 %run %t 2>&1 | FileCheck %s

// CHECK: Set 14 miss rate: 0 / {{.*}} = 0.00%
// CHECK-NOT: Set

int main() {
  return 0;
}
