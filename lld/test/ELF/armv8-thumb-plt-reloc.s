// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -arm-add-build-attributes --arch=thumb --mcpu=cortex-m33 %p/Inputs/arm-plt-reloc.s -o %t1
// RUN: llvm-mc -filetype=obj -arm-add-build-attributes --arch=thumb --mcpu=cortex-m33 %s -o %t2
// RUN: ld.lld %t1 %t2 -o %t
// RUN: llvm-objdump --no-print-imm-hex -d %t | FileCheck %s
// RUN: ld.lld -shared %t1 %t2 -o %t.so
// RUN: llvm-objdump --no-print-imm-hex -d %t.so | FileCheck --check-prefix=DSO %s
// RUN: llvm-readobj -S -r %t.so | FileCheck -check-prefix=DSOREL %s

// RUN: llvm-mc -filetype=obj -arm-add-build-attributes --arch=thumbeb --mcpu=cortex-m33 %p/Inputs/arm-plt-reloc.s -o %t1.be
// RUN: llvm-mc -filetype=obj -arm-add-build-attributes --arch=thumbeb --mcpu=cortex-m33 %s -o %t2.be
// RUN: ld.lld %t1.be %t2.be -o %t.be
// RUN: llvm-objdump --no-print-imm-hex -d %t.be | FileCheck %s
// RUN: ld.lld -shared %t1.be %t2.be -o %t.so.be
// RUN: llvm-objdump --no-print-imm-hex -d %t.so.be | FileCheck --check-prefix=DSO %s
// RUN: llvm-readobj -S -r %t.so.be | FileCheck -check-prefix=DSOREL %s

// RUN: ld.lld --be8 %t1.be %t2.be -o %t.be
// RUN: llvm-objdump --no-print-imm-hex -d %t.be | FileCheck %s
// RUN: ld.lld --be8 -shared %t1.be %t2.be -o %t.so.be
// RUN: llvm-objdump --no-print-imm-hex -d %t.so.be | FileCheck --check-prefix=DSO %s
// RUN: llvm-readobj -S -r %t.so.be | FileCheck -check-prefix=DSOREL %s

// Test PLT entry generation
 .text
 .align 2
 .globl _start
 .type  _start,%function
_start:
// FIXME, interworking is only supported for BL via BLX at the moment, when
// interworking thunks are available for b.w and b<cond>.w this can be altered
// to test the different forms of interworking.
 bl func1
 bl func2
 bl func3

// Executable, expect no PLT
// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: <func1>:
// CHECK-NEXT:   200b4: 4770    bx      lr
// CHECK: <func2>:
// CHECK-NEXT:   200b6: 4770    bx      lr
// CHECK: <func3>:
// CHECK-NEXT:   200b8: 4770    bx      lr
// CHECK-NEXT:   200ba: d4d4 
// CHECK: <_start>:
// CHECK-NEXT:   200bc: f7ff fffa       bl      0x200b4 <func1>
// CHECK-NEXT:   200c0: f7ff fff9       bl      0x200b6 <func2>
// CHECK-NEXT:   200c4: f7ff fff8       bl      0x200b8 <func3>

// Expect PLT entries as symbols can be preempted
// .text is Thumb and .plt is ARM, llvm-objdump can currently only disassemble
// as ARM or Thumb. Work around by disassembling twice.
// DSO: Disassembly of section .text:
// DSO-EMPTY:
// DSO-NEXT: <func1>:
// DSO-NEXT:     10214:     4770    bx      lr
// DSO: <func2>:
// DSO-NEXT:     10216:     4770    bx      lr
// DSO: <func3>:
// DSO-NEXT:     10218:     4770    bx      lr
// DSO-NEXT:     1021a:     d4d4 
// DSO: <_start>:
// 0x10250 = PLT func1
// DSO-NEXT:     1021c:     f000 f818       bl     0x10250
// 0x10260 = PLT func2
// DSO-NEXT:     10220:     f000 f81e       bl     0x10260
// 0x10270 = PLT func3
// DSO-NEXT:     10224:     f000 f824       bl     0x10270
// DSO: Disassembly of section .plt:
// DSO-EMPTY:
// DSO-NEXT: <.plt>:
// DSO-NEXT:     10230: b500          push    {lr}
// DSO-NEXT:     10232: f8df e008     ldr.w   lr, [pc, #8]
// DSO-NEXT:     10236: 44fe          add     lr, pc
// DSO-NEXT:     10238: f85e ff08     ldr     pc, [lr, #8]!
// 0x20098 = .got.plt (0x302D8) - pc (0x10238 = .plt + 8) - 8
// DSO-NEXT:     1023c: {{.*}}        .word   0x00020098
// DSO-NEXT:     10240: d4 d4 d4 d4   .word   0xd4d4d4d4
// DSO-NEXT:     10244: d4 d4 d4 d4   .word   0xd4d4d4d4
// DSO-NEXT:     10248: d4 d4 d4 d4   .word   0xd4d4d4d4
// DSO-NEXT:     1024c: d4 d4 d4 d4   .word   0xd4d4d4d4

// 136 + 2 << 16 + 0x1025c = 0x302e4 = got entry 1
// DSO-NEXT:     10250:       f240 0c88     movw    r12, #136
// DSO-NEXT:     10254:       f2c0 0c02     movt    r12, #2
// DSO-NEXT:     10258:       44fc          add     r12, pc
// DSO-NEXT:     1025a:       f8dc f000     ldr.w   pc, [r12]
// DSO-NEXT:     1025e:       e7fc          b       0x1025a
// 124 + 2 << 16 + 0x1026c = 0x302e8 = got entry 2
// DSO-NEXT:     10260:       f240 0c7c     movw    r12, #124
// DSO-NEXT:     10264:       f2c0 0c02     movt    r12, #2
// DSO-NEXT:     10268:       44fc          add     r12, pc
// DSO-NEXT:     1026a:       f8dc f000     ldr.w   pc, [r12]
// DSO-NEXT:     1026e:       e7fc          b       0x1026a
// 112 + 2 << 16 + 0x1027c = 0x302ec = got entry 3
// DSO-NEXT:     10270:       f240 0c70     movw    r12, #112
// DSO-NEXT:     10274:       f2c0 0c02     movt    r12, #2
// DSO-NEXT:     10278:       44fc          add     r12, pc
// DSO-NEXT:     1027a:       f8dc f000     ldr.w   pc, [r12]
// DSO-NEXT:     1027e:       e7fc          b       0x1027a

// DSOREL:    Name: .got.plt
// DSOREL-NEXT:    Type: SHT_PROGBITS
// DSOREL-NEXT:    Flags [
// DSOREL-NEXT:      SHF_ALLOC
// DSOREL-NEXT:      SHF_WRITE
// DSOREL-NEXT:    ]
// DSOREL-NEXT:    Address: 0x302D8
// DSOREL-NEXT:    Offset:
// DSOREL-NEXT:    Size: 24
// DSOREL-NEXT:    Link:
// DSOREL-NEXT:    Info:
// DSOREL-NEXT:    AddressAlignment: 4
// DSOREL-NEXT:    EntrySize:
// DSOREL:  Relocations [
// DSOREL-NEXT:  Section (5) .rel.plt {
// DSOREL-NEXT:    0x302E4 R_ARM_JUMP_SLOT func1
// DSOREL-NEXT:    0x302E8 R_ARM_JUMP_SLOT func2
// DSOREL-NEXT:    0x302EC R_ARM_JUMP_SLOT func3
