## Check that llvm-bolt prints data embedded in code.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -fno-PIC -no-pie %t.o -o %t.exe -nostdlib \
# RUN:    -fuse-ld=lld -Wl,-q

## Check disassembly of BOLT input.
# RUN: llvm-objdump %t.exe -d | FileCheck %s

# RUN: llvm-bolt %t.exe -o %t.bolt --print-disasm | FileCheck %s

.text
.balign 4

.global _start
.type _start, %function
_start:
  mov x0, #0x0
  .word 0x4f82e010
  ret
  .byte 0x0, 0xff, 0x42
# CHECK-LABEL: _start
# CHECK:        mov x0, #0x0
# CHECK-NEXT:   .word 0x4f82e010
# CHECK-NEXT:   ret
# CHECK-NEXT:   .short 0xff00
# CHECK-NEXT:   .byte 0x42
.size _start, .-_start

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
