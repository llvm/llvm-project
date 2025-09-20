# This test is to ensure that we query data marker symbols to avoid
# misidentifying constant data island symbol as extra entry point.

# RUN: %clang %cflags %s -o %t.so -Wl,-q -Wl,--init=_bar -Wl,--fini=_bar
# RUN: llvm-bolt %t.so -o %t.instr.so

  .text
  .global _start
  .type _start, %function
_start:
  ret

  .text
  .global _foo
  .type _foo, %function
_foo:
  cbz x1, _foo_2
_foo_1:
  add x1, x2, x0
  b _foo
_foo_2:
  ret

# None of these constant island symbols should be identified as extra entry
# point for function `_foo'.
  .align 4
_const1: .short 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80
_const2: .short 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0
_const3: .short 0x04, 0x08, 0x0c, 0x20, 0x60, 0x80, 0xa0, 0xc0

  .text
  .global _bar
  .type _bar, %function
_bar:
  ret

  # Dummy relocation to force relocation mode
  .reloc 0, R_AARCH64_NONE
