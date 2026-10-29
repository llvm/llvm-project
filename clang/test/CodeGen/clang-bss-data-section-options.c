// Test -fclang-bss-section= and -fclang-data-section=, which behave as if the
// translation unit started with the matching '#pragma clang section'.
//
// Like the pragma, these options do not emit an IR 'section' clause. They
// attach "bss-section"/"data-section" string attributes to the global, and the
// backend then picks the one matching the global's kind. Only
// __attribute__((section())) emits a 'section' clause directly.

// RUN: %clang_cc1 -emit-llvm -triple arm-none-eabi \
// RUN:   -fclang-bss-section=.my_bss -fclang-data-section=.my_data \
// RUN:   -o - %s | FileCheck %s --check-prefix=BOTH

// Each option works on its own.
// RUN: %clang_cc1 -emit-llvm -triple arm-none-eabi \
// RUN:   -fclang-bss-section=.my_bss -o - %s | FileCheck %s --check-prefix=BSS
// RUN: %clang_cc1 -emit-llvm -triple arm-none-eabi \
// RUN:   -fclang-data-section=.my_data -o - %s | FileCheck %s --check-prefix=DATA

// Without the options nothing is placed in a non-default section.
// RUN: %clang_cc1 -emit-llvm -triple arm-none-eabi -o - %s \
// RUN:   | FileCheck %s --check-prefix=NONE

int zeroed;
int initialized = 1;
const int constant = 1;

// BOTH: @initialized ={{.*}} global i32 1, align 4 #[[BOTH_ATTR:[0-9]+]]
// BOTH: @constant ={{.*}} constant i32 1, align 4 #[[BOTH_ATTR]]
// BOTH: @zeroed ={{.*}} global i32 0, align 4 #[[BOTH_ATTR]]
//
// The group carries no "rodata-section", so 'constant' is left in the default
// read-only section even though it shares the attribute group.
// BOTH: attributes #[[BOTH_ATTR]] = { "bss-section"=".my_bss" "data-section"=".my_data" }

// BSS: @initialized ={{.*}} global i32 1, align 4 #[[BSS_ATTR:[0-9]+]]
// BSS: @zeroed ={{.*}} global i32 0, align 4 #[[BSS_ATTR]]
// BSS: attributes #[[BSS_ATTR]] = { "bss-section"=".my_bss" }
// BSS-NOT: data-section

// DATA: @initialized ={{.*}} global i32 1, align 4 #[[DATA_ATTR:[0-9]+]]
// DATA: @zeroed ={{.*}} global i32 0, align 4 #[[DATA_ATTR]]
// DATA: attributes #[[DATA_ATTR]] = { "data-section"=".my_data" }
// DATA-NOT: bss-section

// NONE: @initialized ={{.*}} global i32 1, align 4{{$}}
// NONE: @constant ={{.*}} constant i32 1, align 4{{$}}
// NONE: @zeroed ={{.*}} global i32 0, align 4{{$}}
// NONE-NOT: bss-section
// NONE-NOT: data-section