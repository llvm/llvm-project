// REQUIRES: aarch64
// RUN: rm -rf %t && split-file %s %t && cd %t

//--- a.s
.section .tbss,"awT",@nobits
.global a
a:
.xword 0

//--- ok.s
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-linux -mattr=+pauth ok.s -o ok.o
// RUN: ld.lld -shared ok.o -o ok.so
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn ok.so | \
// RUN:   FileCheck -DP=20 -DA=896 -DB=912 -DC=928 %s
// RUN: llvm-readobj -r -x .got ok.so | FileCheck --check-prefix=REL \
// RUN:   -DP1=20 -DA1=380 -DB1=390 -DC1=3A0 -DP2=020 -DA2=380 -DB2=390 -DC2=3a0 %s

// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-linux -mattr=+pauth a.s -o a.so.o
// RUN: ld.lld -shared a.so.o -soname=so -o a.so
// RUN: ld.lld ok.o a.so -o ok
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn ok | \
// RUN:   FileCheck -DP=220 -DA=936 -DB=952 -DC=968 %s
// RUN: llvm-readobj -r -x .got ok | FileCheck --check-prefix=REL \
// RUN:   -DP1=220 -DA1=3A8 -DB1=3B8 -DC1=3C8 -DP2=220 -DA2=3a8 -DB2=3b8 -DC2=3c8 %s

        .text
        adrp    x0, :tlsdesc_auth:a
        ldr     x16, [x0, :tlsdesc_auth_lo12:a]
        add     x0, x0, :tlsdesc_auth_lo12:a
        blraa   x16, x0

// CHECK:      adrp    x0, 0x[[P]]000
// CHECK-NEXT: ldr     x16, [x0, #[[A]]]
// CHECK-NEXT: add     x0, x0, #[[A]]
// CHECK-NEXT: blraa   x16, x0

/// Create relocation against local TLS symbols where linker should
/// create target specific dynamic TLSDESC relocation where addend is
/// the symbol VMA in tls block.

        adrp    x0, :tlsdesc_auth:local1
        ldr     x16, [x0, :tlsdesc_auth_lo12:local1]
        add     x0, x0, :tlsdesc_auth_lo12:local1
        blraa   x16, x0

// CHECK:      adrp    x0, 0x[[P]]000
// CHECK-NEXT: ldr     x16, [x0, #[[B]]]
// CHECK-NEXT: add     x0, x0, #[[B]]
// CHECK-NEXT: blraa   x16, x0

        adrp    x0, :tlsdesc_auth:local2
        ldr     x16, [x0, :tlsdesc_auth_lo12:local2]
        add     x0, x0, :tlsdesc_auth_lo12:local2
        blraa   x16, x0

// CHECK:      adrp    x0, 0x[[P]]000
// CHECK-NEXT: ldr     x16, [x0, #[[C]]]
// CHECK-NEXT: add     x0, x0, #[[C]]
// CHECK-NEXT: blraa   x16, x0

        .section .tbss,"awT",@nobits
        .type   local1,@object
        .p2align 2
local1:
        .word   0
        .size   local1, 4

        .type   local2,@object
        .p2align 3
local2:
        .xword  0
        .size   local2, 8


// R_AARCH64_AUTH_TLSDESC - 0x0 -> start of tls block
// R_AARCH64_AUTH_TLSDESC - 0x8 -> align (sizeof (local1), 8)

// REL:      Relocations [
// REL-NEXT:   Section (5) .rela.dyn {
// REL-NEXT:     0x[[P1]][[B1]] R_AARCH64_AUTH_TLSDESC - 0x0
// REL-NEXT:     0x[[P1]][[C1]] R_AARCH64_AUTH_TLSDESC - 0x8
// REL-NEXT:     0x[[P1]][[A1]] R_AARCH64_AUTH_TLSDESC a 0x0
// REL-NEXT:   }
// REL-NEXT: ]

// REL:      Hex dump of section '.got':
// REL-NEXT: 0x00[[P2]][[A2]] 00000000 00000080 00000000 000000a0
// REL-NEXT: 0x00[[P2]][[B2]] 00000000 00000080 00000000 000000a0
// REL-NEXT: 0x00[[P2]][[C2]] 00000000 00000080 00000000 000000a0
///                                          ^^
///                                          0b10000000 bit 63 address diversity = true, bits 61..60 key = IA
///                                                            ^^
///                                                            0b10100000 bit 63 address diversity = true, bits 61..60 key = DA

//--- err1.s
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-linux -mattr=+pauth err1.s -o err1.o
// RUN: not ld.lld -shared err1.o 2>&1 | FileCheck --check-prefix=ERR1 --implicit-check-not=error: %s
// ERR1: error: both AUTH and non-AUTH TLSDESC entries for 'a' requested, but only one type of TLSDESC entry per symbol is supported
        .text
        adrp    x0, :tlsdesc_auth:a
        ldr     x16, [x0, :tlsdesc_auth_lo12:a]
        add     x0, x0, :tlsdesc_auth_lo12:a
        blraa   x16, x0

        adrp    x0, :tlsdesc:a
        ldr     x1, [x0, :tlsdesc_lo12:a]
        add     x0, x0, :tlsdesc_lo12:a
        blr     x1

//--- err2.s
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-linux -mattr=+pauth err2.s -o err2.o
// RUN: not ld.lld -shared err2.o 2>&1 | FileCheck --check-prefix=ERR2 --implicit-check-not=error: %s
// ERR2: error: both AUTH and non-AUTH TLSDESC entries for 'a' requested, but only one type of TLSDESC entry per symbol is supported
        .text
        adrp    x0, :tlsdesc:a
        ldr     x1, [x0, :tlsdesc_lo12:a]
        add     x0, x0, :tlsdesc_lo12:a
        blr     x1

        adrp    x0, :tlsdesc_auth:a
        ldr     x16, [x0, :tlsdesc_auth_lo12:a]
        add     x0, x0, :tlsdesc_auth_lo12:a
        blraa   x16, x0

//--- err3.s
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-linux -mattr=+pauth err3.s -o err3.o
// RUN: not ld.lld -shared err3.o 2>&1 | FileCheck --check-prefix=ERR3 --implicit-check-not=error: %s
// ERR3: error: both AUTH and non-AUTH TLSDESC entries for 'a' requested, but only one type of TLSDESC entry per symbol is supported
        .text
        adrp    x0, :tlsdesc_auth:a
        ldr     x16, [x0, :tlsdesc_auth_lo12:a]
        add     x0, x0, :tlsdesc_auth_lo12:a
        .tlsdesccall a
        blraa   x16, x0
