// RUN: %clangxx %s -o %t
// RUN: %env_tool_opts=enable_symbolizer_markup=1 %run %t 2>&1 | FileCheck %s

// REQUIRES: linux
#include <sanitizer/common_interface_defs.h>

void Bar() { __sanitizer_print_stack_trace(); }

void Foo() {
  Bar();
  return;
}

void Baz() { __sanitizer_print_stack_trace(); }

int main() {
  Foo();
  Baz();
  return 0;
}

// COM: For element syntax see: https://llvm.org/docs/SymbolizerMarkupFormat.html
// COM: OPEN is {{{ and CLOSE is }}}

// CHECK: [[OPEN:{{{]]reset[[CLOSE:}}}]]
// CHECK: [[OPEN]]module:[[MOD_ID:[0-9]+]]:{{.+}}:elf:{{[0-9a-fA-F]+}}[[CLOSE]]
// CHECK: [[OPEN]]mmap:{{0x[0-9a-fA-F]+:0x[0-9a-fA-F]+}}:load:[[MOD_ID]]:{{r[wx]{0,2}:0x[0-9a-fA-F]+}}[[CLOSE]]
// CHECK: [[OPEN]]bt:0:0x{{[0-9a-fA-F]+}}[[CLOSE]]
// CHECK-NEXT: [[OPEN]]bt:1:0x{{[0-9a-fA-F]+}}[[CLOSE]]
// CHECK-NEXT: [[OPEN]]bt:2:0x{{[0-9a-fA-F]+}}[[CLOSE]]

// COM: Emitting a second backtrace should not emit contextual elements in this case.
// CHECK-NOT: [[OPEN:{{{]]reset[[CLOSE:}}}]]
// CHECK-NOT: [[OPEN]]module:[[MOD_ID:[0-9]+]]:{{.+}}:elf:{{[0-9a-fA-F]+}}[[CLOSE]]
// CHECK-NOT: [[OPEN]]mmap:{{0x[0-9a-fA-F]+:0x[0-9a-fA-F]+}}:load:[[MOD_ID]]:{{r[wx]{0,2}:0x[0-9a-fA-F]+}}[[CLOSE]]

// CHECK: [[OPEN]]bt:0:0x{{[0-9a-fA-F]+}}[[CLOSE]]
// CHECK-NEXT: [[OPEN]]bt:1:0x{{[0-9a-fA-F]+}}[[CLOSE]]
