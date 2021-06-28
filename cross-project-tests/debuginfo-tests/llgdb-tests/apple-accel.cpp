// REQUIRES: system-darwin
// Test that clang produces the __apple accelerator tables,
// e.g., __apple_types, correctly.
// These sections are going to be retired in DWARF 5, so we hardcode
// the DWARF version in the tests.
// RUN: %clang %s %target_itanium_abi_host_triple -gdwarf-2 -O0 -c -g -o %t-ex
// RUN: llvm-objdump --section-headers %t-ex | FileCheck %s
// RUN: %clang %s %target_itanium_abi_host_triple -gdwarf-4 -O0 -c -g -o %t-ex
// RUN: llvm-objdump --section-headers %t-ex | FileCheck %s

// A function in a different section forces the compiler to create the
// __debug_ranges section.
__attribute__((section("1,__text_foo"))) void foo() {}
int main (int argc, char const *argv[]) { return argc; }

// CHECK-DAG: __debug_abbrev
// CHECK-DAG: __debug_info
// CHECK-DAG: __debug_str
// CHECK-DAG: __debug_ranges
// CHECK-DAG: __apple_names
// CHECK-DAG: __apple_objc
// CHECK-DAG: __apple_namespac
// CHECK-DAG: __apple_types
