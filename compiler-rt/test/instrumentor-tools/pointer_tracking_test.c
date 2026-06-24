// Test pointer tracking functionality
//
// This test verifies that the pointer tracking runtime correctly tracks
// allocations, replacements, and usage patterns (read/write).
//
// RUN: %clangxx -O0 -g -mllvm -enable-instrumentor -mllvm -instrumentor-read-config-files=%config_dir/pointer-tracking/pointer_tracking_config.json %s -L%lib_dir -l%pointer_tracking_lib -o %t
// RUN: %t | FileCheck %s
//
// CHECK: Pointer Tracking Statistics
// CHECK: Total allocations tracked: {{[0-9]+}}
// CHECK: Read-only:
// CHECK: Write-only:
// CHECK: Read and Write:
// CHECK: Longest Unused Times
// CHECK: Longest First Use Times

#include <stdio.h>

// Global variables to test global instrumentation
int global_read_only = 42;
int global_write_only = 0;
int global_read_write = 100;

// Function with various allocation patterns
void test_allocations(void) {
  // Alloca: read-only
  int read_only_local = 10;
  int temp = read_only_local; // Read
  (void)temp;

  // Alloca: write-only
  int write_only_local;
  write_only_local = 20; // Write

  // Alloca: read and write
  int read_write_local = 30;
  read_write_local += 5;  // Read and write

  // Alloca: unused
  int unused_local;
  (void)unused_local;
}

void test_globals(void) {
  // Read from global
  int temp = global_read_only;
  (void)temp;

  // Write to global
  global_write_only = 50;

  // Read and write global
  global_read_write += 10;
}

void test_arrays(void) {
  int array[10];

  // Write to array elements
  for (int i = 0; i < 10; i++) {
    array[i] = i * 2;
  }

  // Read from array elements
  int sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += array[i];
  }

  // TODO: This won't work until we decompose escaping pointers.
  // printf("Array sum: %d\n", sum);
}

void test_delayed_use(void) {
  // Allocation with delayed first use
  int delayed = 5;

  // Do some work to create time gap
  volatile int busy_work = 0;
  for (int i = 0; i < 1000; i++) {
    busy_work += i;
  }

  // Now use the delayed allocation
  int result = delayed * 2;
  (void)result;
}

int main(void) {
  test_allocations();
  test_globals();
  test_arrays();
  test_delayed_use();

  // TODO: This won't work until we decompose escaping pointers.
  // printf("Pointer tracking test complete\n");
  return 0;
}
