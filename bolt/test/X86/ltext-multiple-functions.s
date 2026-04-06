## Test that BOLT handles multiple functions split across .text and .ltext
## sections, assigning each to the correct output section.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe --emit-relocs -e _start
# RUN: llvm-bolt %t.exe -o %t.bolt --reorder-blocks=none 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-readelf -S %t.bolt | FileCheck %s --check-prefix=CHECK-SEC
# RUN: llvm-objdump --syms %t.bolt | FileCheck %s --check-prefix=CHECK-SYMS

# CHECK-BOLT: large code model detected

## Both .ltext (with large flag) and .text (without) must exist in output.
# CHECK-SEC: .ltext {{.*}} AXl
# CHECK-SEC: .text  {{.*}} AX

## Functions from .ltext stay in .ltext; functions from .text stay in .text.
# CHECK-SYMS-DAG: .ltext{{.*}} large_foo
# CHECK-SYMS-DAG: .ltext{{.*}} large_bar
# CHECK-SYMS-DAG: .text{{.*}} _start
# CHECK-SYMS-DAG: .text{{.*}} small_helper

  .text
  .globl _start
  .type _start, @function
_start:
  call large_foo
  call small_helper
  xorl %eax, %eax
  retq
  .size _start, .-_start

  .globl small_helper
  .type small_helper, @function
small_helper:
  movl $1, %eax
  retq
  .size small_helper, .-small_helper

  .section .ltext,"axl",@progbits
  .globl large_foo
  .type large_foo, @function
large_foo:
  pushq %rbp
  movq %rsp, %rbp
  call large_bar
  popq %rbp
  retq
  .size large_foo, .-large_foo

  .globl large_bar
  .type large_bar, @function
large_bar:
  movl $42, %eax
  retq
  .size large_bar, .-large_bar
