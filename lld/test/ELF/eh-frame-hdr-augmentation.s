// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld --hash-style=sysv --eh-frame-hdr %t.o -o %t -shared
// RUN: llvm-objdump --dwarf=frames %t | FileCheck %s

// CHECK: .eh_frame contents:

// CHECK:      00000000 0000001c 00000000 CIE
// CHECK-NEXT:   Version:                       1
// CHECK-NEXT:   Augmentation:             "zPLR"
// CHECK-NEXT:   Code alignment factor: 1
// CHECK-NEXT:   Data alignment factor: -8
// CHECK-NEXT:   Return address column: 16
// CHECK-NEXT:   Personality Address: 0000000000001029
// CHECK-NEXT:   Augmentation data:

// CHECK:      DW_CFA_def_cfa:  reg7 +8
// CHECK-NEXT: DW_CFA_offset:   reg16 -8
// CHECK-NEXT: DW_CFA_nop:
// CHECK-NEXT: DW_CFA_nop:

// CHECK:      00000020 00000014 00000024 FDE cie=00000000 pc=00001014...00001014
// CHECK-NEXT:   LSDA Address: 000000000000100b
// CHECK-NEXT:   DW_CFA_nop:
// CHECK-NEXT:   DW_CFA_nop:
// CHECK-NEXT:   DW_CFA_nop:

        .cfi_startproc
        .cfi_personality 0x9b, g
        .cfi_lsda 0x1b, h
        .cfi_endproc

        .global g
        .hidden g
g:

        .global h
        .hidden h
h:

