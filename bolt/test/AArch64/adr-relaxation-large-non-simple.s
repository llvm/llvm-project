## Check that an ADR targeting the same fragment is not relaxed in a large
## non-simple function. BOLT preserves the layout of non-simple functions, so
## the ADR displacement cannot change even when the function is larger than the
## instruction's 1MiB range.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=false

  .text
  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
.Ladr:
  adr x1, .Ladr
  br x0

  // Make the function's code larger than 1MiB. The unknown indirect branch
  // makes the function non-simple, while the self-referential ADR remains in
  // range.
  .rept 0x40000
  nop
  .endr
  ret
  .cfi_endproc
  .size _start, .-_start

  // Force BOLT's relocation mode.
  .reloc 0, R_AARCH64_NONE
