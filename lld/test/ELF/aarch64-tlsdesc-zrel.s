/// Check that we write addends for AArch64 TLSDESC relocations with -z rel
/// See https://bugs.llvm.org/show_bug.cgi?id=47009
// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: ld.lld -shared %t.o -o %t-rela.so
// RUN: llvm-readobj -W -r -x .got %t-rela.so | FileCheck %s --check-prefixes=RELA,RELA-NO-ADDENDS
// RUN: ld.lld -shared %t.o -o %t-rela-addends.so --apply-dynamic-relocs
// RUN: llvm-readobj -W -r -x .got %t-rela-addends.so | FileCheck %s --check-prefixes=RELA,RELA-WITH-ADDENDS

// RELA:       Relocations [
// RELA-NEXT:   Section (5) .rela.dyn {
// RELA-NEXT:     0x20340 R_AARCH64_TLSDESC - 0x0
// RELA-NEXT:     0x20350 R_AARCH64_TLSDESC - 0x4
// RELA-NEXT:   }
// RELA-NEXT: ]
// RELA-NEXT:              Hex dump of section '.got':
// RELA-NEXT:              0x00020340 00000000 00000000 00000000 00000000
// RELA-NO-ADDENDS-NEXT:   0x00020350 00000000 00000000 00000000 00000000
// RELA-WITH-ADDENDS-NEXT: 0x00020350 00000000 00000000 04000000 00000000
///                Addend 0x4 for R_AARCH64_TLSDESC -----^
// RELA-EMPTY:

// RUN: ld.lld -shared %t.o -o %t-rel.so -z rel
// RUN: llvm-readobj -W -r -x .got %t-rel.so | FileCheck %s --check-prefix=REL
// REL:       Relocations [
// REL-NEXT:   Section (5) .rel.dyn {
// REL-NEXT:     0x20330 R_AARCH64_TLSDESC -{{$}}
// REL-NEXT:     0x20340 R_AARCH64_TLSDESC -{{$}}
// REL-NEXT:   }
// REL-NEXT: ]
// REL-NEXT: Hex dump of section '.got':
// REL-NEXT: 0x00020330 00000000 00000000 00000000 00000000
// REL-NEXT: 0x00020340 00000000 00000000 04000000 00000000
///  Addend 0x4 for R_AARCH64_TLSDESC -----^
// REL-EMPTY:

        .text
foo:
        adrp    x0, :tlsdesc:x
        ldr     x1, [x0, :tlsdesc_lo12:x]
        add     x0, x0, :tlsdesc_lo12:x
        .tlsdesccall x
        blr     x1
        adrp    x0, :tlsdesc:y
        ldr     x1, [x0, :tlsdesc_lo12:y]
        add     x0, x0, :tlsdesc_lo12:y
        .tlsdesccall y
        blr     x1
        ret

        .section        .tbss,"awT",@nobits
        .p2align        2
        .hidden x
x:
        .word   0

        .p2align        2
        .hidden y
y:
        .word   0
