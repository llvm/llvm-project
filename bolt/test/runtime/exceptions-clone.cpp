// Test that clones at origin work correctly with C++ exception handling.
//
// The test creates two caller functions:
// - caller_optimized: will be relocated/optimized by BOLT
// - caller_skipped: will be skipped (not relocated) by BOLT
//
// With --no-scan, BOLT won't scan unrelocated code for references, so:
// - caller_skipped's call to catch_exception remains at original address
//   (clone)
// - caller_optimized's call to catch_exception is rewritten to new address
//
// This tests that both versions of catch_exception work correctly.

// clang-format off

// REQUIRES: system-linux

// RUN: %clangxx %cxxflags -no-pie -O1 -fno-inline %s -Wl,-q -o %t

// Get the original catch_exception address for later verification.
// RUN: llvm-nm %t | grep " catch_exception$" | cut -d' ' -f1 > %t.orig_addr

// Run BOLT with clone-at-origin.
// RUN: llvm-bolt %t -o %t.bolt \
// RUN:   --skip-funcs=caller_skipped \
// RUN:   --no-scan \
// RUN:   --clone-at-origin \
// RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-BOLT

// CHECK-BOLT: BOLT-INFO: created {{[0-9]+}} clones at origin

// Verify the clone symbol exists in the output binary.
// RUN: llvm-nm %t.bolt | grep ".clone.0" | FileCheck %s --check-prefix=CHECK-SYM

// CHECK-SYM: .clone.0

// Verify that the clone symbol address matches the original function address.
// RUN: llvm-nm %t.bolt | grep "catch_exception.clone.0" | cut -d' ' -f1 > %t.clone_addr
// RUN: diff %t.orig_addr %t.clone_addr

// The relocated function should be at a DIFFERENT address than the original.
// RUN: llvm-nm %t.bolt | grep " catch_exception$" | cut -d' ' -f1 > %t.bolt_addr
// RUN: not diff %t.orig_addr %t.bolt_addr

// Verify caller_skipped calls the clone (at original address).
// RUN: llvm-objdump -d %t.bolt --disassemble-symbols=caller_skipped \
// RUN:   | FileCheck %s --check-prefix=CHECK-SKIPPED-CALL

// CHECK-SKIPPED-CALL: <caller_skipped>:
// CHECK-SKIPPED-CALL: jmp{{.*}}<catch_exception.clone.0>

// Verify caller_optimized calls the relocated function (not the clone).
// RUN: llvm-objdump -d %t.bolt --disassemble-symbols=caller_optimized \
// RUN:   | FileCheck %s --check-prefix=CHECK-OPTIMIZED-CALL

// CHECK-OPTIMIZED-CALL: <caller_optimized>:
// CHECK-OPTIMIZED-CALL: jmp{{.*}}<catch_exception>

// Run the bolted binary - both callers should work:
// - caller_skipped calls original code (clone) at original address
// - caller_optimized calls relocated code at new address
// RUN: %t.bolt 2>&1 | FileCheck %s --check-prefix=CHECK-EXEC

// CHECK-EXEC: Testing exception handling with clones
// CHECK-EXEC: caller_skipped -> catch_exception (original/clone):
// CHECK-EXEC-NEXT: caught: 42
// CHECK-EXEC: caller_optimized -> catch_exception (relocated):
// CHECK-EXEC-NEXT: caught: 42
// CHECK-EXEC: All exceptions handled correctly

#include <cstdio>
#include <exception>

// This function will be cloned. It catches an exception.
// Using extern "C" to avoid name mangling for easier testing.
extern "C" __attribute__((noinline)) void catch_exception() {
  try {
    throw 42;
  } catch (int e) {
    printf("caught: %d\n", e);
  }
}

// This caller will be SKIPPED by BOLT (not relocated).
// Its call to catch_exception will remain pointing to the original address.
// With --no-scan, this call won't be updated, so it calls the clone.
extern "C" __attribute__((noinline)) void caller_skipped() {
  printf("caller_skipped -> catch_exception (original/clone):\n");
  catch_exception();
}

// This caller will be OPTIMIZED by BOLT (relocated).
// Its call to catch_exception will be rewritten to the new address.
extern "C" __attribute__((noinline)) void caller_optimized() {
  printf("caller_optimized -> catch_exception (relocated):\n");
  catch_exception();
}

int main() {
  printf("Testing exception handling with clones\n");

  caller_skipped();
  caller_optimized();

  printf("All exceptions handled correctly\n");
  return 0;
}
