// REQUIRES: aarch64-registered-target

// RUN: %clang --target=aarch64-unknown-fuchsia -c -g -O1 -gdwarf-4 %s -o - | \
// RUN: llvm-dwarfdump --debug-info - | FileCheck %s --check-prefix=CHECK

struct Base {
  virtual void foo();
} *B;

void bar() { B->foo(); }

// CHECK: DW_TAG_compile_unit
// CHECK:   DW_TAG_subprogram
// CHECK:     DW_AT_GNU_all_call_sites	(true)
// CHECK:     DW_AT_linkage_name	("_Z3barv")
// CHECK:     DW_TAG_GNU_call_site
// CHECK:       DW_AT_GNU_call_site_target_clobbered
// CHECK:       DW_AT_GNU_tail_call	(true)
