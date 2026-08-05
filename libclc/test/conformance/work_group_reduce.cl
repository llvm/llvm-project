// RUN: %libclc-compile-and-run --threads-x 64 %t

__kernel void test(void) {
  uint lid = get_local_id(0);
  uint n = get_local_size(0);
  uint sum = work_group_reduce_add(lid);
  if (lid == 0 && sum != n * (n - 1) / 2)
    __builtin_verbose_trap("libclc", "work_group_reduce_add mismatch");
}
