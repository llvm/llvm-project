// Test the histogram support in memprof using the text format output.
// Shadow memory counters per object are limited to 8b. In memory counters
// aggregating counts across multiple objects are 64b.

// RUN: %clangxx_memprof -O0 -mllvm -memprof-histogram -mllvm -memprof-use-callbacks=true %s -o %t
// RUN: %env_memprof_opts=print_text=1:histogram=1:log_path=stdout %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

int main() {
  // Allocate memory that will create a histogram
  char *buffer = (char *)malloc(1024);
  if (!buffer)
    return 1;

  for (int i = 0; i < 10; ++i) {
    // Access every 8th byte (since shadow granularity is 8b.
    buffer[i * 8] = 'A';
  }

  for (int j = 0; j < 200; ++j) {
    buffer[8] = 'B'; // Count = previous count + 200
  }

  for (int j = 0; j < 400; ++j) {
    buffer[16] = 'B'; // Count is saturated at 255
  }

  // Free the memory to trigger MIB creation with histogram
  free(buffer);

  printf("Test completed successfully\n");
  return 0;
}

// CHECK: AccessCountHistogram[128]: 1 201 255 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// CHECK: Test completed successfully
