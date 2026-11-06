// A '#pragma clang section' in the source overrides -fclang-bss-section= and
// -fclang-data-section= from its point of appearance onwards.
//
// As with the pragma, the section names are carried by "bss-section"/
// "data-section" string attributes on the global rather than by an IR
// 'section' clause; only __attribute__((section())) emits the latter.

// RUN: %clang_cc1 -emit-llvm -triple arm-none-eabi \
// RUN:   -fclang-bss-section=.opt_bss -fclang-data-section=.opt_data \
// RUN:   -o - %s | FileCheck %s

// An explicit __attribute__((section())) still wins over both.
__attribute__((section(".attr_data"))) int attributed = 1;

int before_zeroed;
int before_initialized = 1;

#pragma clang section bss=".pragma_bss" data=".pragma_data"
int during_zeroed;
int during_initialized = 1;

// Clearing the pragma reverts to the default section, not to the option.
#pragma clang section bss="" data=""
int after_zeroed;
int after_initialized = 1;

// The attribute wins outright: a 'section' clause and no section attributes.
// CHECK: @attributed ={{.*}} global i32 1, section ".attr_data", align 4{{$}}
// CHECK: @before_initialized ={{.*}} global i32 1, align 4 #[[OPT:[0-9]+]]
// CHECK: @during_initialized ={{.*}} global i32 1, align 4 #[[PRAGMA:[0-9]+]]
// CHECK: @after_initialized ={{.*}} global i32 1, align 4{{$}}
// CHECK: @before_zeroed ={{.*}} global i32 0, align 4 #[[OPT]]
// CHECK: @during_zeroed ={{.*}} global i32 0, align 4 #[[PRAGMA]]
// CHECK: @after_zeroed ={{.*}} global i32 0, align 4{{$}}

// CHECK: attributes #[[OPT]] = { "bss-section"=".opt_bss" "data-section"=".opt_data" }
// CHECK: attributes #[[PRAGMA]] = { "bss-section"=".pragma_bss" "data-section"=".pragma_data" }