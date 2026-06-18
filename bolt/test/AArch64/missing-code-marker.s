## Check that llvm-bolt is able to recover a missing code marker.

# RUN: %clang %cflags %s -o %t.exe -nostdlib -fuse-ld=lld -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck %s

# CHECK: BOLT-WARNING: function symbol foo lacks code marker

.text
.balign 4

.word 0

## Function foo starts immediately after a data object and does not have
## a matching "$x" symbol to indicate the start of code.
.global foo
.type foo, %function
foo:
  .word 0xd65f03c0
.size foo, .-foo

.global _start
.type _start, %function
_start:
  bl foo
  ret
.size _start, .-_start
