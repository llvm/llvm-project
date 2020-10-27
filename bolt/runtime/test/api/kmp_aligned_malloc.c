// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include "omp_testsuite.h"

int alignments[] = {64, 128, 256, 512, 1024, 2048, 4096};

unsigned aligned_by(uint64_t addr) {
    uint64_t alignment = 1;
    while((addr & (alignment-1)) == 0) {
        alignment <<= 1;
    }
    return (alignment >> 1);
}

int test_kmp_aligned_malloc()
{
  int err = 0;
  #pragma omp parallel shared(err)
  {
    int i;
    int* ptr;
    uint64_t addr;
    int tid = omp_get_thread_num();

    for(i = 0; i < sizeof(alignments)/sizeof(int); i++) {
      int alignment = alignments[i];
      // allocate 64 bytes with 64-byte alignment
      // allocate 128 bytes with 128-byte alignment, etc.
      ptr = (int*)kmp_aligned_malloc(alignment, alignment);
      addr = (uint64_t)ptr;
      if(addr & (alignment-1)) {
        printf("thread %d: addr = %p (aligned to %u bytes) but expected "
               " alignment = %d\n", tid, ptr, aligned_by(addr), alignment);
        err = 1;
      }
      kmp_free(ptr);
    }

    ptr = kmp_aligned_malloc(128, 127);
    if (ptr != NULL) {
      printf("thread %d: kmp_aligned_malloc() didn't return NULL when "
             "alignment was not power of 2\n", tid);
      err = 1;
    }
  } /* end of parallel */
  return !err;
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_kmp_aligned_malloc()) {
      num_failed++;
    }
  }
  return num_failed;
}
