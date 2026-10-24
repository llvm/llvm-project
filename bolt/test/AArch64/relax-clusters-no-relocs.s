# Check that clustered relaxation rejects binaries without code relocations.

# RUN: %clang %cflags %s -o %t.exe
# RUN: not llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck %s
# RUN: llvm-bolt %t.exe -o %t.compact.bolt --compact-code-model --relax-with-clusters=false

# CHECK: BOLT-ERROR: clustered relaxation requires relocations against code; link with --emit-relocs or pass --relax-with-clusters=false

  .text
  .globl _start
  .type _start, %function
_start:
  ret
  .size _start, .-_start
