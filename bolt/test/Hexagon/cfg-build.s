## Verify that BOLT can build a CFG for simple Hexagon functions.

# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-linux-musl %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe --emit-relocs -e _start
# RUN: llvm-bolt %t.exe --print-cfg --print-only=_start -o /dev/null \
# RUN:   > %t.log 2>&1
# RUN: FileCheck %s --input-file=%t.log

# CHECK: Binary Function "_start" after building cfg
# CHECK: .LBB00 (2 instructions, align : 1)
# CHECK-NEXT:   Entry Point
# CHECK-NEXT:     {{.*}}: call foo
# CHECK:          {{.*}}: jumpr r31

  .text
  .globl _start
  .type _start,@function
  .p2align 4
_start:
    call foo
    jumpr r31
  .size _start, .-_start

  .globl foo
  .type foo,@function
  .p2align 4
foo:
    r0 = #0
    jumpr r31
  .size foo, .-foo
