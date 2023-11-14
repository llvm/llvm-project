# REQUIRES: x86

# Test merging code sections with data sections.

# RUN: llvm-mc -triple x86_64-windows-msvc %s -filetype=obj -o %t.obj
# RUN: lld-link -machine:amd64 -dll -noentry -out:%t.dll %t.obj -merge:.testx=.testd -merge:.testx2=.testbss -merge:.testd2=.testx3 -merge:.testbss2=.testx4

# RUN: llvm-readobj --sections %t.dll | FileCheck %s
# CHECK:      Sections [
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Number: 1
# CHECK-NEXT:     Name: .testbss (2E 74 65 73 74 62 73 73)
# CHECK-NEXT:     VirtualSize: 0x18
# CHECK-NEXT:     VirtualAddress: 0x1000
# CHECK-NEXT:     RawDataSize: 512
# CHECK-NEXT:     PointerToRawData: 0x400
# CHECK-NEXT:     PointerToRelocations: 0x0
# CHECK-NEXT:     PointerToLineNumbers: 0x0
# CHECK-NEXT:     RelocationCount: 0
# CHECK-NEXT:     LineNumberCount: 0
# CHECK-NEXT:     Characteristics [ (0xC0000080)
# CHECK-NEXT:       IMAGE_SCN_CNT_UNINITIALIZED_DATA (0x80)
# CHECK-NEXT:       IMAGE_SCN_MEM_READ (0x40000000)
# CHECK-NEXT:       IMAGE_SCN_MEM_WRITE (0x80000000)
# CHECK-NEXT:     ]
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Number: 2
# CHECK-NEXT:     Name: .testd (2E 74 65 73 74 64 00 00)
# CHECK-NEXT:     VirtualSize: 0x18
# CHECK-NEXT:     VirtualAddress: 0x2000
# CHECK-NEXT:     RawDataSize: 512
# CHECK-NEXT:     PointerToRawData: 0x600
# CHECK-NEXT:     PointerToRelocations: 0x0
# CHECK-NEXT:     PointerToLineNumbers: 0x0
# CHECK-NEXT:     RelocationCount: 0
# CHECK-NEXT:     LineNumberCount: 0
# CHECK-NEXT:     Characteristics [ (0x40000040)
# CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA (0x40)
# CHECK-NEXT:       IMAGE_SCN_MEM_READ (0x40000000)
# CHECK-NEXT:     ]
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Number: 3
# CHECK-NEXT:     Name: .testx3 (2E 74 65 73 74 78 33 00)
# CHECK-NEXT:     VirtualSize: 0x12
# CHECK-NEXT:     VirtualAddress: 0x3000
# CHECK-NEXT:     RawDataSize: 512
# CHECK-NEXT:     PointerToRawData: 0x800
# CHECK-NEXT:     PointerToRelocations: 0x0
# CHECK-NEXT:     PointerToLineNumbers: 0x0
# CHECK-NEXT:     RelocationCount: 0
# CHECK-NEXT:     LineNumberCount: 0
# CHECK-NEXT:     Characteristics [ (0x60000020)
# CHECK-NEXT:       IMAGE_SCN_CNT_CODE (0x20)
# CHECK-NEXT:       IMAGE_SCN_MEM_EXECUTE (0x20000000)
# CHECK-NEXT:       IMAGE_SCN_MEM_READ (0x40000000)
# CHECK-NEXT:     ]
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Number: 4
# CHECK-NEXT:     Name: .testx4 (2E 74 65 73 74 78 34 00)
# CHECK-NEXT:     VirtualSize: 0x14
# CHECK-NEXT:     VirtualAddress: 0x4000
# CHECK-NEXT:     RawDataSize: 512
# CHECK-NEXT:     PointerToRawData: 0xA00
# CHECK-NEXT:     PointerToRelocations: 0x0
# CHECK-NEXT:     PointerToLineNumbers: 0x0
# CHECK-NEXT:     RelocationCount: 0
# CHECK-NEXT:     LineNumberCount: 0
# CHECK-NEXT:     Characteristics [ (0x60000020)
# CHECK-NEXT:       IMAGE_SCN_CNT_CODE (0x20)
# CHECK-NEXT:       IMAGE_SCN_MEM_EXECUTE (0x20000000)
# CHECK-NEXT:       IMAGE_SCN_MEM_READ (0x40000000)
# CHECK-NEXT:     ]
# CHECK-NEXT:   }
# CHECK-NEXT: ]

# RUN: llvm-objdump -d %t.dll | FileCheck -check-prefix=DISASM %s
# DISASM:      Disassembly of section .testx3:
# DISASM-EMPTY:
# DISASM-NEXT: 0000000180003000 <.testx3>:
# DISASM-NEXT: 180003000: 48 c7 c0 03 00 00 00         movq    $0x3, %rax
# DISASM-NEXT: 180003007: c3                           retq
# DISASM-NEXT: 180003008: cc                           int3
# DISASM-NEXT: 180003009: cc                           int3
# DISASM-NEXT: 18000300a: cc                           int3
# DISASM-NEXT: 18000300b: cc                           int3
# DISASM-NEXT: 18000300c: cc                           int3
# DISASM-NEXT: 18000300d: cc                           int3
# DISASM-NEXT: 18000300e: cc                           int3
# DISASM-NEXT: 18000300f: cc                           int3
# DISASM-NEXT: 180003010: 02 00                        addb    (%rax), %al
# DISASM-EMPTY:
# DISASM-NEXT: Disassembly of section .testx4:
# DISASM-EMPTY:
# DISASM-NEXT: 0000000180004000 <.testx4>:
# DISASM-NEXT: 180004000: 48 c7 c0 04 00 00 00         movq    $0x4, %rax
# DISASM-NEXT: 180004007: c3                           retq
# DISASM-NEXT: 180004008: cc                           int3
# DISASM-NEXT: 180004009: cc                           int3
# DISASM-NEXT: 18000400a: cc                           int3
# DISASM-NEXT: 18000400b: cc                           int3
# DISASM-NEXT: 18000400c: cc                           int3
# DISASM-NEXT: 18000400d: cc                           int3
# DISASM-NEXT: 18000400e: cc                           int3
# DISASM-NEXT: 18000400f: cc                           int3
# DISASM-NEXT: 180004010: 00 00                        addb    %al, (%rax)
# DISASM-NEXT: 180004012: 00 00                        addb    %al, (%rax)


        .section .testx, "xr"
        .p2align 4
        movq $1, %rax
        retq

        .section .testx2, "xr"
        .p2align 4
        movq $2, %rax
        retq

        .section .testd, "dr"
        .p2align 4
        .word 1

        .section .testbss, "b"
        .p2align 4
        .skip 4

        .section .testx3, "xr"
        .p2align 4
        movq $3, %rax
        retq

        .section .testx4, "xr"
        .p2align 4
        movq $4, %rax
        retq

        .section .testd2, "dr"
        .p2align 4
        .word 2

        .section .testbss2, "b"
        .p2align 4
        .skip 4
