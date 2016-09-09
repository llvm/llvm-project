// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld -static %t.o -o %tout
// RUN: llvm-objdump -triple armv7a-none-linux-gnueabi -d %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-readobj -r -symbols -sections %tout | FileCheck %s
// REQUIRES: arm
 .syntax unified
 .text
 .type foo STT_GNU_IFUNC
 .globl foo
foo:
 bx lr

 .type bar STT_GNU_IFUNC
 .globl bar
bar:
 bx lr

 .globl _start
_start:
 bl foo
 bl bar
 movw r0,:lower16:__rel_iplt_start
 movt r0,:upper16:__rel_iplt_start
 movw r0,:lower16:__rel_iplt_end
 movt r0,:upper16:__rel_iplt_end

// CHECK:      Sections [
// CHECK:       Section {
// CHECK:       Index: 1
// CHECK-NEXT:  Name: .rel.plt
// CHECK-NEXT:  Type: SHT_REL
// CHECK-NEXT:  Flags [
// CHECK-NEXT:    SHF_ALLOC
// CHECK-NEXT:  ]
// CHECK-NEXT:  Address: [[REL:.*]]
// CHECK-NEXT:  Offset:
// CHECK-NEXT:   Size: 16
// CHECK-NEXT:   Link:
// CHECK-NEXT:   Info:
// CHECK-NEXT:   AddressAlignment: 4
// CHECK-NEXT:   EntrySize: 8
// CHECK-NEXT:  }
// CHECK: Relocations [
// CHECK-NEXT:  Section (1) .rel.plt {
// CHECK-NEXT:    0x1200C R_ARM_IRELATIVE
// CHECK-NEXT:    0x12010 R_ARM_IRELATIVE
// CHECK-NEXT:  }
// CHECK-NEXT:]
// CHECK:  Symbols [
// CHECK:   Symbol {
// CHECK:         Name: __rel_iplt_end
// CHECK-NEXT:    Value: 0x100E4
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other [
// CHECK-NEXT:      STV_HIDDEN
// CHECK-NEXT:    ]
// CHECK-NEXT:    Section: .rel.plt
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: __rel_iplt_start
// CHECK-NEXT:    Value: 0x100D4
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other [
// CHECK-NEXT:      STV_HIDDEN
// CHECK-NEXT:    ]
// CHECK-NEXT:    Section: .rel.plt
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: _start (38)
// CHECK-NEXT:    Value: 0x11008
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other:
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: bar
// CHECK-NEXT:    Value: 0x11004
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: GNU_IFunc
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: foo
// CHECK-NEXT:    Value: 0x11000
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: GNU_IFunc
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }

// DISASM:      Disassembly of section .text:
// DISASM-NEXT: foo:
// DISASM-NEXT:   11000:       1e ff 2f e1     bx      lr
// DISASM: bar:
// DISASM-NEXT:   11004:       1e ff 2f e1     bx      lr
// DISASM: _start:
// DISASM-NEXT:   11008:       09 00 00 eb     bl      #36
// DISASM-NEXT:   1100c:       0c 00 00 eb     bl      #48
// DISASM-NEXT:   11010:       d4 00 00 e3     movw    r0, #212
// DISASM-NEXT:   11014:       01 00 40 e3     movt    r0, #1
// r0 = 212 + 1 * 65536 = 100D4 = __rel_iplt_start
// DISASM-NEXT:   11018:       e4 00 00 e3     movw    r0, #228
// DISASM-NEXT:   1101c:       01 00 40 e3     movt    r0, #1
// r1 = 228 + 1 * 65536 = 100E4 = __rel_iplt_end
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-NEXT: .plt:
// DISASM-NEXT:   11020:       04 e0 2d e5     str     lr, [sp, #-4]!
// DISASM-NEXT:   11024:       04 e0 9f e5     ldr     lr, [pc, #4]
// DISASM-NEXT:   11028:       0e e0 8f e0     add     lr, pc, lr
// DISASM-NEXT:   1102c:       08 f0 be e5     ldr     pc, [lr, #8]!
// 0x0fd0 + 0x11028 + 0x8 = 0x12000
// DISASM-NEXT:   11030:       d0 0f 00 00
// DISASM-NEXT:   11034:       04 c0 9f e5     ldr     r12, [pc, #4]
// DISASM-NEXT:   11038:       0f c0 8c e0     add     r12, r12, pc
// DISASM-NEXT:   1103c:       00 f0 9c e5     ldr     pc, [r12]
// 0x0fcc + 0x11038 + 0x8 = 0x1200C
// DISASM-NEXT:   11040:       cc 0f 00 00
// DISASM-NEXT:   11044:       04 c0 9f e5     ldr     r12, [pc, #4]
// DISASM-NEXT:   11048:       0f c0 8c e0     add     r12, r12, pc
// DISASM-NEXT:   1104c:       00 f0 9c e5     ldr     pc, [r12]
// 0x0fc0 + 0x11048 + 0x8 = 0x12010
// DISASM-NEXT:   11050:       c0 0f 00 00
