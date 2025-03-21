// REQUIRES: aarch64
// RUN: rm -rf %t && split-file %s %t && cd %t

// RUN: llvm-mc -filetype=obj -triple=aarch64 start.s -o start.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 xo-same-section.s -o xo-same-section.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 rx-same-section.s -o rx-same-section.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 xo-different-section.s -o xo-different-section.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 rx-different-section.s -o rx-different-section.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 %p/Inputs/plt-aarch64.s -o plt.o
// RUN: ld.lld -shared plt.o -soname=t2.so -o plt.so
// RUN: ld.lld start.o xo-same-section.o plt.so -o xo-same-section
// RUN: ld.lld start.o rx-same-section.o plt.so -o rx-same-section
// RUN: ld.lld start.o xo-different-section.o plt.so -o xo-different-section
// RUN: ld.lld start.o rx-different-section.o plt.so -o rx-different-section
// RUN: llvm-readelf -S -l xo-same-section | FileCheck --check-prefix=CHECK-XO %s
// RUN: llvm-readelf -S -l rx-same-section | FileCheck --check-prefix=CHECK-RX %s
// RUN: llvm-readelf -S -l xo-different-section | FileCheck --check-prefix=CHECK-XO %s
// RUN: llvm-readelf -S -l rx-different-section | FileCheck --check-prefix=CHECK-RX %s
// RUN: llvm-objdump -d --no-show-raw-insn xo-same-section | FileCheck --check-prefix=DISASM %s
// RUN: llvm-objdump -d --no-show-raw-insn rx-same-section | FileCheck --check-prefix=DISASM %s
// RUN: llvm-objdump -d --no-show-raw-insn xo-different-section | FileCheck --check-prefix=DISASM %s
// RUN: llvm-objdump -d --no-show-raw-insn rx-different-section | FileCheck --check-prefix=DISASM %s

///          Name Type     Address          Off    Size   ES Flg Lk Inf Al
// CHECK-XO: .plt PROGBITS 00000000002102e0 0002e0 000040 00 AXy  0   0 16
// CHECK-RX: .plt PROGBITS 00000000002102e0 0002e0 000040 00  AX  0   0 16

/// The address of .plt above should be within this program header.
///          Type Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
// CHECK-XO: LOAD 0x0002c8 0x00000000002102c8 0x00000000002102c8 0x000058 0x000058   E 0x10000
// CHECK-RX: LOAD 0x0002c8 0x00000000002102c8 0x00000000002102c8 0x000058 0x000058 R E 0x10000

// DISASM-LABEL: Disassembly of section .plt:
// DISASM-LABEL: <.plt>:
// DISASM-NEXT:  2102e0: stp  x16, x30, [sp, #-0x10]!
// DISASM-NEXT:          adrp x16, 0x230000 <weak+0x230000>
// DISASM-NEXT:          ldr  x17, [x16, #0x400]
// DISASM-NEXT:          add  x16, x16, #0x400
// DISASM-NEXT:          br   x17
// DISASM-NEXT:          nop
// DISASM-NEXT:          nop
// DISASM-NEXT:          nop

// DISASM-LABEL: <bar@plt>:
// DISASM-NEXT:  210300: adrp x16, 0x230000 <weak+0x230000>
// DISASM-NEXT:          ldr  x17, [x16, #0x408]
// DISASM-NEXT:          add  x16, x16, #0x408
// DISASM-NEXT:          br   x17

// DISASM-LABEL: <weak@plt>:
// DISASM-NEXT:  210310: adrp x16, 0x230000 <weak+0x230000>
// DISASM-NEXT:          ldr  x17, [x16, #0x410]
// DISASM-NEXT:          add  x16, x16, #0x410
// DISASM-NEXT:          br   x17

//--- start.s
.section .text,"axy",@progbits,unique,0
.global _start, foo, bar
.weak weak
_start:
  bl foo
  bl bar
  bl weak
  ret

//--- xo-same-section.s
.section .text,"axy",@progbits,unique,0
.global foo
foo:
  ret

//--- rx-same-section.s
.section .text,"ax",@progbits,unique,0
.global foo
foo:
  ret

//--- xo-different-section.s
.section .foo,"axy",@progbits,unique,0
.global foo
foo:
  ret

//--- rx-different-section.s
.section .foo,"ax",@progbits,unique,0
.global foo
foo:
  ret
