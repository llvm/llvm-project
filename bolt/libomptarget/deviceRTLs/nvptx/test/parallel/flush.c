// RUN: %compile-run-and-check

#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int data, out, flag = 0;
#pragma omp target parallel num_threads(64) map(tofrom                         \
                                                : out, flag) map(to            \
                                                                 : data)
  {
    if (omp_get_thread_num() == 0) {
      /* Write to the data buffer that will be read by thread */
      data = 42;
/* Flush data to thread 32 */
#pragma omp flush(data)
      /* Set flag to release thread 32 */
#pragma omp atomic write
      flag = 1;
    } else if (omp_get_thread_num() == 32) {
      /* Loop until we see the update to the flag */
      int val;
      do {
#pragma omp atomic read
        val = flag;
      } while (val < 1);
      out = data;
#pragma omp flush(out)
    }
  }
  // CHECK: out=42.
  /* Value of out will be 42 */
  printf("out=%d.\n", out);
  return !(out == 42);
}
