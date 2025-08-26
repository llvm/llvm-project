# This test is to verify that BOLT won't take a label pointing to constant
# island as a secondary entry point (function `_start` doesn't have ELF size
# set originally) and the function won't otherwise be mistaken as non-simple.

# RUN: %clang %cflags -pie %s -o %t.so -Wl,-q -Wl,--init=_foo -Wl,--fini=_foo
# RUN: llvm-bolt %t.so -o %t.bolt.so --print-cfg 2>&1 | FileCheck %s
# CHECK-NOT: BOLT-WARNING: reference in the middle of instruction detected \
# CHECK-NOT:   function _start at offset 0x{{[0-9a-f]+}}
# CHECK: Binary Function "_start" after building cfg

  .text

  .global _foo
  .type _foo, %function
_foo:
  ret

  .global _start
  .type _start, %function
_start:
  b _foo

  .balign 16
_random_consts:
  .long 0x12345678
  .long 0x90abcdef

  .global _bar
  .type _bar, %function
_bar:
  ret

  # Dummy relocation to force relocation mode
  .reloc 0, R_AARCH64_NONE
