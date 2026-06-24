// Test basic pointer tracking functionality
//
// This test verifies that the pointer tracking runtime works with simple
// stack allocations and basic read/write operations.
//
// RUN: %clangxx -O0 -g -mllvm -enable-instrumentor -mllvm -instrumentor-read-config-files=%config_dir/pointer-tracking/pointer_tracking_config.json %s -L%lib_dir -l%pointer_tracking_lib -o %t
// RUN: %t | FileCheck %s
//
// CHECK: Pointer Tracking Statistics
// CHECK: Total allocations tracked: 5
// CHECK: Usage Patterns:
// CHECK: Read-only: 0
// CHECK: Write-only: 3
// CHECK: Read and Write: 2

#include <stdio.h>

int main(void) {
  // Test 1: Read-only allocation
  int read_only = 42;
  int copy = read_only;
  (void)copy;

  // Test 2: Write-only allocation
  int write_only;
  write_only = 10;
  (void)write_only;

  // Test 3: Read-write allocation
  int read_write = 5;
  read_write = read_write * 2;
  (void)read_write;

  // TODO: This needs escaping pointer support
  //  printf("Simple test complete\n");
  return 0;
}
