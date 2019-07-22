// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %p/Inputs/arm-plt-reloc.s -o %t1
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t2
// RUN: ld.lld %t1 %t2 -o %t
// RUN: llvm-objdump -triple=armv7a-none-linux-gnueabi -d --no-show-raw-insn %t | FileCheck %s
// RUN: ld.lld --hash-style=sysv -shared %t1 %t2 -o %t3
// RUN: llvm-objdump -triple=armv7a-none-linux-gnueabi -d --no-show-raw-insn %t3 | FileCheck -check-prefix=DSO %s
// RUN: llvm-readobj -S -r %t3 | FileCheck -check-prefix=DSOREL %s
//
// Test PLT entry generation
 .syntax unified
 .text
 .align 2
 .globl _start
 .type  _start,%function
_start:
 b func1
 bl func2
 beq func3

// Executable, expect no PLT
// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: func1:
// CHECK-NEXT:   11000:       bx      lr
// CHECK: func2:
// CHECK-NEXT:   11004:       bx      lr
// CHECK: func3:
// CHECK-NEXT:   11008:       bx      lr
// CHECK: _start:
// CHECK-NEXT:   1100c:       b       #-20 <func1>
// CHECK-NEXT:   11010:       bl      #-20 <func2>
// CHECK-NEXT:   11014:       beq     #-20 <func3>

// Expect PLT entries as symbols can be preempted
// The .got.plt and .plt displacement is small so we can use small PLT entries.
// DSO: Disassembly of section .text:
// DSO-EMPTY:
// DSO-NEXT: func1:
// DSO-NEXT:     1000:       bx      lr
// DSO: func2:
// DSO-NEXT:     1004:       bx      lr
// DSO: func3:
// DSO-NEXT:     1008:       bx      lr
// DSO: _start:
// S(0x1040) - P(0x100c) + A(-8) = 0x2c = 32
// DSO-NEXT:     100c:       b       #44
// S(0x1050) - P(0x1010) + A(-8) = 0x38 = 56
// DSO-NEXT:     1010:       bl      #56
// S(0x10160) - P(0x1014) + A(-8) = 0x44 = 68
// DSO-NEXT:     1014:       beq     #68
// DSO-EMPTY:
// DSO-NEXT: Disassembly of section .plt:
// DSO-EMPTY:
// DSO-NEXT: $a:
// DSO-NEXT:     1020:       str     lr, [sp, #-4]!
// (0x1024 + 8) + (0 RoR 12) + 4096 + (0xfdc) = 0x3008 = .got.plt[3]
// DSO-NEXT:     1024:       add     lr, pc, #0, #12
// DSO-NEXT:     1028:       add     lr, lr, #4096
// DSO-NEXT:     102c:       ldr     pc, [lr, #4060]!
// DSO: $d:
// DSO-NEXT:     1030:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO-NEXT:     1034:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO-NEXT:     1038:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO-NEXT:     103c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO: $a:
// (0x1040 + 8) + (0 RoR 12) + 4096 + (0xfc4) = 0x300c
// DSO-NEXT:     1040:       add     r12, pc, #0, #12
// DSO-NEXT:     1044:       add     r12, r12, #4096
// DSO-NEXT:     1048:       ldr     pc, [r12, #4036]!
// DSO: $d:
// DSO-NEXT:     104c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO: $a:
// (0x1050 + 8) + (0 RoR 12) + 4096 + (0xfb8) = 0x3010
// DSO-NEXT:     1050:       add     r12, pc, #0, #12
// DSO-NEXT:     1054:       add     r12, r12, #4096
// DSO-NEXT:     1058:       ldr     pc, [r12, #4024]!
// DSO: $d:
// DSO-NEXT:     105c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO: $a:
// (0x1060 + 8) + (0 RoR 12) + 4096 + (0xfac) = 0x3014
// DSO-NEXT:     1060:       add     r12, pc, #0, #12
// DSO-NEXT:     1064:       add     r12, r12, #4096
// DSO-NEXT:     1068:       ldr     pc, [r12, #4012]!
// DSO: $d:
// DSO-NEXT:     106c:       d4 d4 d4 d4     .word   0xd4d4d4d4


// DSOREL:    Name: .got.plt
// DSOREL-NEXT:    Type: SHT_PROGBITS
// DSOREL-NEXT:    Flags [
// DSOREL-NEXT:      SHF_ALLOC
// DSOREL-NEXT:      SHF_WRITE
// DSOREL-NEXT:    ]
// DSOREL-NEXT:    Address: 0x3000
// DSOREL-NEXT:    Offset:
// DSOREL-NEXT:    Size: 24
// DSOREL-NEXT:    Link:
// DSOREL-NEXT:    Info:
// DSOREL-NEXT:    AddressAlignment: 4
// DSOREL-NEXT:    EntrySize:
// DSOREL:  Relocations [
// DSOREL-NEXT:  Section {{.*}} .rel.plt {
// DSOREL-NEXT:    0x300C R_ARM_JUMP_SLOT func1 0x0
// DSOREL-NEXT:    0x3010 R_ARM_JUMP_SLOT func2 0x0
// DSOREL-NEXT:    0x3014 R_ARM_JUMP_SLOT func3 0x0

// Test a large separation between the .plt and .got.plt
// The .got.plt and .plt displacement is large but still within the range
// of the short plt sequence.
// RUN: echo "SECTIONS { \
// RUN:       .text 0x1000 : { *(.text) } \
// RUN:       .plt  0x2000 : { *(.plt) *(.plt.*) } \
// RUN:       .got.plt 0x1100000 : { *(.got.plt) } \
// RUN:       }" > %t.script
// RUN: ld.lld --hash-style=sysv --script %t.script -shared %t1 %t2 -o %t4
// RUN: llvm-objdump -triple=armv7a-none-linux-gnueabi -d --no-show-raw-insn %t4 | FileCheck --check-prefix=CHECKHIGH %s
// RUN: llvm-readobj -S -r %t4 | FileCheck --check-prefix=DSORELHIGH %s

// CHECKHIGH: Disassembly of section .text:
// CHECKHIGH-EMPTY:
// CHECKHIGH-NEXT: func1:
// CHECKHIGH-NEXT:     1000:       bx      lr
// CHECKHIGH: func2:
// CHECKHIGH-NEXT:     1004:       bx      lr
// CHECKHIGH: func3:
// CHECKHIGH-NEXT:     1008:       bx      lr
// CHECKHIGH: _start:
// CHECKHIGH-NEXT:     100c:       b       #4108 <$a>
// CHECKHIGH-NEXT:     1010:       bl      #4120 <$a>
// CHECKHIGH-NEXT:     1014:       beq     #4132 <$a>
// CHECKHIGH-EMPTY:
// CHECKHIGH-NEXT: Disassembly of section .plt:
// CHECKHIGH-EMPTY:
// CHECKHIGH-NEXT: $a:
// CHECKHIGH-NEXT:     2000:       str     lr, [sp, #-4]!
// CHECKHIGH-NEXT:     2004:       add     lr, pc, #16, #12
// CHECKHIGH-NEXT:     2008:       add     lr, lr, #1036288
// CHECKHIGH-NEXT:     200c:       ldr     pc, [lr, #4092]!
// CHECKHIGH: $d:
// CHECKHIGH-NEXT:     2010:       d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECKHIGH-NEXT:     2014:       d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECKHIGH-NEXT:     2018:       d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECKHIGH-NEXT:     201c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECKHIGH: $a:
// CHECKHIGH-NEXT:     2020:       add     r12, pc, #16, #12
// CHECKHIGH-NEXT:     2024:       add     r12, r12, #1036288
// CHECKHIGH-NEXT:     2028:       ldr     pc, [r12, #4068]!
// CHECKHIGH: $d:
// CHECKHIGH-NEXT:     202c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECKHIGH: $a:
// CHECKHIGH-NEXT:     2030:       add     r12, pc, #16, #12
// CHECKHIGH-NEXT:     2034:       add     r12, r12, #1036288
// CHECKHIGH-NEXT:     2038:       ldr     pc, [r12, #4056]!
// CHECKHIGH: $d:
// CHECKHIGH-NEXT:     203c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECKHIGH: $a:
// CHECKHIGH-NEXT:     2040:       add     r12, pc, #16, #12
// CHECKHIGH-NEXT:     2044:       add     r12, r12, #1036288
// CHECKHIGH-NEXT:     2048:       ldr     pc, [r12, #4044]!
// CHECKHIGH: $d:
// CHECKHIGH-NEXT:     204c:       d4 d4 d4 d4     .word   0xd4d4d4d4

// DSORELHIGH:     Name: .got.plt
// DSORELHIGH-NEXT:     Type: SHT_PROGBITS
// DSORELHIGH-NEXT:     Flags [
// DSORELHIGH-NEXT:       SHF_ALLOC
// DSORELHIGH-NEXT:       SHF_WRITE
// DSORELHIGH-NEXT:     ]
// DSORELHIGH-NEXT:     Address: 0x1100000
// DSORELHIGH: Relocations [
// DSORELHIGH-NEXT:   Section {{.*}} .rel.plt {
// DSORELHIGH-NEXT:     0x110000C R_ARM_JUMP_SLOT func1 0x0
// DSORELHIGH-NEXT:     0x1100010 R_ARM_JUMP_SLOT func2 0x0
// DSORELHIGH-NEXT:     0x1100014 R_ARM_JUMP_SLOT func3 0x0

// Test a very large separation between the .plt and .got.plt so we must use
// large plt entries that do not have any range restriction.
// RUN: echo "SECTIONS { \
// RUN:       .text 0x1000 : { *(.text) } \
// RUN:       .plt  0x2000 : { *(.plt) *(.plt.*) } \
// RUN:       .got.plt 0x11111100 : { *(.got.plt) } \
// RUN:       }" > %t2.script
// RUN: ld.lld --hash-style=sysv --script %t2.script -shared %t1 %t2 -o %t5
// RUN: llvm-objdump -triple=armv7a-none-linux-gnueabi -d --no-show-raw-insn %t5 | FileCheck --check-prefix=CHECKLONG %s
// RUN: llvm-readobj -S -r %t5 | FileCheck --check-prefix=DSORELLONG %s

// CHECKLONG: Disassembly of section .text:
// CHECKLONG-EMPTY:
// CHECKLONG-NEXT: func1:
// CHECKLONG-NEXT:     1000:       bx      lr
// CHECKLONG: func2:
// CHECKLONG-NEXT:     1004:       bx      lr
// CHECKLONG: func3:
// CHECKLONG-NEXT:     1008:       bx      lr
// CHECKLONG: _start:
// CHECKLONG-NEXT:     100c:       b       #4108 <$a>
// CHECKLONG-NEXT:     1010:       bl      #4120 <$a>
// CHECKLONG-NEXT:     1014:       beq     #4132 <$a>
// CHECKLONG-EMPTY:
// CHECKLONG-NEXT: Disassembly of section .plt:
// CHECKLONG-EMPTY:
// CHECKLONG-NEXT: $a:
// CHECKLONG-NEXT:     2000:       str     lr, [sp, #-4]!
// CHECKLONG-NEXT:     2004:       ldr     lr, [pc, #4]
// CHECKLONG-NEXT:     2008:       add     lr, pc, lr
// CHECKLONG-NEXT:     200c:       ldr     pc, [lr, #8]!
// CHECKLONG: $d:
// CHECKLONG-NEXT:     2010:       f0 f0 10 11     .word   0x1110f0f0
// CHECKLONG-NEXT:     2014:       d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECKLONG-NEXT:     2018:       d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECKLONG-NEXT:     201c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECKLONG: $a:
// CHECKLONG-NEXT:     2020:       ldr     r12, [pc, #4]
// CHECKLONG-NEXT:     2024:       add     r12, r12, pc
// CHECKLONG-NEXT:     2028:       ldr     pc, [r12]
// CHECKLONG: $d:
// CHECKLONG-NEXT:     202c:       e0 f0 10 11     .word   0x1110f0e0
// CHECKLONG: $a:
// CHECKLONG-NEXT:     2030:       ldr     r12, [pc, #4]
// CHECKLONG-NEXT:     2034:       add     r12, r12, pc
// CHECKLONG-NEXT:     2038:       ldr     pc, [r12]
// CHECKLONG: $d:
// CHECKLONG-NEXT:     203c:       d4 f0 10 11     .word   0x1110f0d4
// CHECKLONG: $a:
// CHECKLONG-NEXT:     2040:       ldr     r12, [pc, #4]
// CHECKLONG-NEXT:     2044:       add     r12, r12, pc
// CHECKLONG-NEXT:     2048:       ldr     pc, [r12]
// CHECKLONG: $d:
// CHECKLONG-NEXT:     204c:       c8 f0 10 11     .word   0x1110f0c8

// DSORELLONG: Name: .got.plt
// DSORELLONG-NEXT:     Type: SHT_PROGBITS
// DSORELLONG-NEXT:     Flags [
// DSORELLONG-NEXT:       SHF_ALLOC
// DSORELLONG-NEXT:       SHF_WRITE
// DSORELLONG-NEXT:     ]
// DSORELLONG-NEXT:     Address: 0x11111100
// DSORELLONG: Relocations [
// DSORELLONG-NEXT:   Section {{.*}} .rel.plt {
// DSORELLONG-NEXT:     0x1111110C R_ARM_JUMP_SLOT func1 0x0
// DSORELLONG-NEXT:     0x11111110 R_ARM_JUMP_SLOT func2 0x0
// DSORELLONG-NEXT:     0x11111114 R_ARM_JUMP_SLOT func3 0x0

// Test a separation between the .plt and .got.plt that is part in range of
// short table entries and part needing long entries. We use the long entries
// only when we need to.
// RUN: echo "SECTIONS { \
// RUN:       .text 0x1000 : { *(.text) } \
// RUN:       .plt  0x2000 : { *(.plt) *(.plt.*) } \
// RUN:       .got.plt 0x8002020 : { *(.got.plt) } \
// RUN:       }" > %t3.script
// RUN: ld.lld --hash-style=sysv --script %t3.script -shared %t1 %t2 -o %t6
// RUN: llvm-objdump -triple=armv7a-none-linux-gnueabi -d --no-show-raw-insn %t6 | FileCheck --check-prefix=CHECKMIX %s
// RUN: llvm-readobj -S -r %t6 | FileCheck --check-prefix=DSORELMIX %s

// CHECKMIX: Disassembly of section .text:
// CHECKMIX-EMPTY:
// CHECKMIX-NEXT: func1:
// CHECKMIX-NEXT:     1000:       bx      lr
// CHECKMIX: func2:
// CHECKMIX-NEXT:     1004:       bx      lr
// CHECKMIX: func3:
// CHECKMIX-NEXT:     1008:       bx      lr
// CHECKMIX: _start:
// CHECKMIX-NEXT:     100c:       b       #4108 <$a>
// CHECKMIX-NEXT:     1010:       bl      #4120 <$a>
// CHECKMIX-NEXT:     1014:       beq     #4132 <$a>
// CHECKMIX-EMPTY:
// CHECKMIX-NEXT: Disassembly of section .plt:
// CHECKMIX-EMPTY:
// CHECKMIX-NEXT: $a:
// CHECKMIX-NEXT:     2000:       str     lr, [sp, #-4]!
// CHECKMIX-NEXT:     2004:       ldr     lr, [pc, #4]
// CHECKMIX-NEXT:     2008:       add     lr, pc, lr
// CHECKMIX-NEXT:     200c:       ldr     pc, [lr, #8]!
// CHECKMIX: $d:
// CHECKMIX-NEXT:     2010:     10 00 00 08     .word   0x08000010
// CHECKMIX-NEXT:     2014:     d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECKMIX-NEXT:     2018:     d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECKMIX-NEXT:     201c:     d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECKMIX: $a:
// CHECKMIX-NEXT:     2020:       ldr     r12, [pc, #4]
// CHECKMIX-NEXT:     2024:       add     r12, r12, pc
// CHECKMIX-NEXT:     2028:       ldr     pc, [r12]
// CHECKMIX: $d:
// CHECKMIX-NEXT:     202c:     00 00 00 08     .word   0x08000000
// CHECKMIX: $a:
// CHECKMIX-NEXT:     2030:       add     r12, pc, #133169152
// CHECKMIX-NEXT:     2034:       add     r12, r12, #1044480
// CHECKMIX-NEXT:     2038:       ldr     pc, [r12, #4088]!
// CHECKMIX: $d:
// CHECKMIX-NEXT:     203c:     d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECKMIX: $a:
// CHECKMIX-NEXT:     2040:       add     r12, pc, #133169152
// CHECKMIX-NEXT:     2044:       add     r12, r12, #1044480
// CHECKMIX-NEXT:     2048:       ldr     pc, [r12, #4076]!
// CHECKMIX: $d:
// CHECKMIX-NEXT:     204c:     d4 d4 d4 d4     .word   0xd4d4d4d4

// DSORELMIX:    Name: .got.plt
// DSORELMIX-NEXT:     Type: SHT_PROGBITS
// DSORELMIX-NEXT:     Flags [
// DSORELMIX-NEXT:       SHF_ALLOC
// DSORELMIX-NEXT:       SHF_WRITE
// DSORELMIX-NEXT:     ]
// DSORELMIX-NEXT:     Address: 0x8002020
// DSORELMIX:   Section {{.*}} .rel.plt {
// DSORELMIX-NEXT:     0x800202C R_ARM_JUMP_SLOT func1 0x0
// DSORELMIX-NEXT:     0x8002030 R_ARM_JUMP_SLOT func2 0x0
// DSORELMIX-NEXT:     0x8002034 R_ARM_JUMP_SLOT func3 0x0
