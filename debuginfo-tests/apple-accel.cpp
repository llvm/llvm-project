// REQUIRES: system-darwin
// Test that clang produces the __apple accelerator tables,
// e.g., __apple_types, correctly.
// RUN: %clang %s %target_itanium_abi_host_triple -O0 -c -g -o %t-ex
// RUN: llvm-objdump -section-headers %t-ex | FileCheck %s

int main (int argc, char const *argv[]) { return argc; }

// CHECK: __debug_str
// CHECK-NEXT: __debug_abbrev
// CHECK-NEXT: __debug_info
// CHECK-NEXT: __debug_ranges
// CHECK-NEXT: __debug_macinfo
// CHECK-NEXT: __apple_names
// CHECK-NEXT: __apple_objc
// CHECK-NEXT: __apple_namespac
// CHECK-NEXT: __apple_types
