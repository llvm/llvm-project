// RUN: %clangxx -O3  %s -o %t
// RUN: %env_tool_opts=strip_path_prefix=/TestCases/ %run %t 2>&1 | FileCheck %s
// RUN: %env_tool_opts=strip_path_prefix=/TestCases/:symbolize_inline_frames=0 \
// RUN:   %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK-NOINLINE

#include <sanitizer/common_interface_defs.h>
#include <stdio.h>
#include <string.h>

char buffer[10000];

__attribute__((noinline)) static void Symbolize() {
  __sanitizer_symbolize_pc(__builtin_return_address(0), "%p %F %L", buffer,
                           sizeof(buffer));
  for (char *p = buffer; strlen(p); p += strlen(p) + 1)
    printf("%s\n", p);
}

// CHECK-NOINLINE: {{0x[0-9a-f]+}} in main symbolize_pc_inline.cc:[[@LINE+2]]
// CHECK: [[ADDR:0x[0-9a-f]+]] in C2 symbolize_pc_inline.cc:[[@LINE+1]]
static inline void C2() { Symbolize(); }

// CHECK: [[ADDR]] in C3 symbolize_pc_inline.cc:[[@LINE+1]]
static inline void C3() { C2(); }

// CHECK: [[ADDR]] in C4 symbolize_pc_inline.cc:[[@LINE+1]]
static inline void C4() { C3(); }

// CHECK: [[ADDR]] in main symbolize_pc_inline.cc:[[@LINE+1]]
int main() { C4(); }
