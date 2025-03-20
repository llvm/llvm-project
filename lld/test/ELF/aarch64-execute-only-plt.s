// REQUIRES: aarch64
// RUN: rm -rf %t && split-file %s %t && cd %t

// RUN: llvm-mc -filetype=obj -triple=aarch64 start.s -o start.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 foo-xo-same-section.s -o foo-xo-same-section.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 foo-rx-same-section.s -o foo-rx-same-section.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 foo-xo-different-section.s -o foo-xo-different-section.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 foo-rx-different-section.s -o foo-rx-different-section.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 %p/Inputs/plt-aarch64.s -o plt.o
// RUN: ld.lld -shared plt.o -soname=t2.so -o plt.so
// RUN: ld.lld start.o foo-xo-same-section.o plt.so -o xo-same-section
// RUN: ld.lld start.o foo-rx-same-section.o plt.so -o rx-same-section
// RUN: ld.lld start.o foo-xo-different-section.o plt.so -o xo-different-section
// RUN: ld.lld start.o foo-rx-different-section.o plt.so -o rx-different-section
// RUN: llvm-readobj -S -l xo-same-section | FileCheck --check-prefix=CHECK-XO %s
// RUN: llvm-readobj -S -l rx-same-section | FileCheck --check-prefix=CHECK-RX %s
// RUN: llvm-readobj -S -l xo-different-section | FileCheck --check-prefix=CHECK-XO %s
// RUN: llvm-readobj -S -l rx-different-section | FileCheck --check-prefix=CHECK-RX %s
// RUN: llvm-objdump -d --no-show-raw-insn xo-same-section | FileCheck --check-prefix=DISASM %s
// RUN: llvm-objdump -d --no-show-raw-insn rx-same-section | FileCheck --check-prefix=DISASM %s
// RUN: llvm-objdump -d --no-show-raw-insn xo-different-section | FileCheck --check-prefix=DISASM %s
// RUN: llvm-objdump -d --no-show-raw-insn rx-different-section | FileCheck --check-prefix=DISASM %s

// CHECK-XO:         Name: .plt
// CHECK-XO-NEXT:    Type: SHT_PROGBITS
// CHECK-XO-NEXT:    Flags [
// CHECK-XO-NEXT:      SHF_AARCH64_PURECODE
// CHECK-XO-NEXT:      SHF_ALLOC
// CHECK-XO-NEXT:      SHF_EXECINSTR
// CHECK-XO-NEXT:    ]
// CHECK-XO-NEXT:    Address: 0x2102E0

/// The address of .plt above should be within this program header.
// CHECK-XO:         VirtualAddress: 0x2102C8
// CHECK-XO-NEXT:    PhysicalAddress: 0x2102C8
// CHECK-XO-NEXT:    FileSize: 88
// CHECK-XO-NEXT:    MemSize: 88
// CHECK-XO-NEXT:    Flags [
// CHECK-XO-NEXT:      PF_X
// CHECK-XO-NEXT:    ]

// CHECK-RX:         Name: .plt
// CHECK-RX-NEXT:    Type: SHT_PROGBITS
// CHECK-RX-NEXT:    Flags [
// CHECK-RX-NEXT:      SHF_ALLOC
// CHECK-RX-NEXT:      SHF_EXECINSTR
// CHECK-RX-NEXT:    ]
// CHECK-RX-NEXT:    Address: 0x2102E0

/// The address of .plt above should be within this program header.
// CHECK-RX:         VirtualAddress: 0x2102C8
// CHECK-RX-NEXT:    PhysicalAddress: 0x2102C8
// CHECK-RX-NEXT:    FileSize: 88
// CHECK-RX-NEXT:    MemSize: 88
// CHECK-RX-NEXT:    Flags [
// CHECK-RX-NEXT:      PF_R
// CHECK-RX-NEXT:      PF_X
// CHECK-RX-NEXT:    ]

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

//--- foo-xo-same-section.s
.section .text,"axy",@progbits,unique,0
.global foo
foo:
  ret

//--- foo-rx-same-section.s
.section .text,"ax",@progbits,unique,0
.global foo
foo:
  ret

//--- foo-xo-different-section.s
.section .foo,"axy",@progbits,unique,0
.global foo
foo:
  ret

//--- foo-rx-different-section.s
.section .foo,"ax",@progbits,unique,0
.global foo
foo:
  ret
