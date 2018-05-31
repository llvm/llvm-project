// REQUIRES: ppc

// RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
// RUN: ld.lld -shared %t.o -o %t.so
// RUN: llvm-readelf -relocations --wide %t.o | FileCheck --check-prefix=InputRelocs %s
// RUN: llvm-readelf -relocations --wide %t.so | FileCheck --check-prefix=OutputRelocs %s
// RUN: llvm-objdump --section-headers %t.so | FileCheck --check-prefix=CheckGot %s
// RUN: llvm-objdump -D %t.so | FileCheck --check-prefix=Dis %s

// RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
// RUN: ld.lld -shared %t.o -o %t.so
// RUN: llvm-readelf -relocations --wide %t.o | FileCheck --check-prefix=InputRelocs %s
// RUN: llvm-readelf -relocations --wide %t.so | FileCheck --check-prefix=OutputRelocs %s
// RUN: llvm-objdump --section-headers %t.so | FileCheck --check-prefix=CheckGot %s
// RUN: llvm-objdump -D %t.so | FileCheck --check-prefix=Dis %s

        .text
        .abiversion 2
        .globl  test
        .p2align        4
        .type   test,@function
test:
.Lfunc_gep0:
        addis 2, 12, .TOC.-.Lfunc_gep0@ha
        addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
        .localentry     test, .Lfunc_lep0-.Lfunc_gep0
        mflr 0
        std 0, 16(1)
        stdu 1, -32(1)
        addis 3, 2, i@got@tlsld@ha
        addi 3, 3, i@got@tlsld@l
        bl __tls_get_addr(i@tlsld)
        nop
        ld 0, 16(1)
        mtlr 0
        blr

        .globl test_hi
        .p2align 4
        .type test_hi,@function
test_hi:
         lis 3, j@got@tlsld@h
         blr

        .globl test_16
        .p2align 4
        .type test_16,@function
test_16:
         li 3, k@got@tlsld
         blr

        .type   i,@object
        .section        .tdata,"awT",@progbits
        .p2align        2
i:
        .long   55
        .size   i, 4

        .type   j,@object
        .section        .tbss,"awT",@nobits
        .p2align        2
j:
        .long   0
        .size   j, 4

        .type   k,@object
        .section        .tdata,"awT",@progbits
        .p2align        3
k:
        .quad   66
        .size   k, 8

// Verify that the input contains all the R_PPC64_GOT_TLSLD16* relocations.
// InputRelocs: Relocation section '.rela.text'
// InputRelocs:     R_PPC64_GOT_TLSLD16_HA 0000000000000000 i + 0
// InputRelocs:     R_PPC64_GOT_TLSLD16_LO 0000000000000000 i + 0
// InputRelocs:     R_PPC64_TLSLD          0000000000000000 i + 0
// InputRelocs:     R_PPC64_GOT_TLSLD16_HI 0000000000000000 j + 0
// InputRelocs:     R_PPC64_GOT_TLSLD16    0000000000000008 k + 0

// The local dynamic version of tls needs to use the same mechanism to look up
// a variables address as general-dynamic. ie a call to __tls_get_addr with the
// address of a tls_index struct as the argument. However for local-dynamic
// variables  all will have the same ti_module, and the offset field is left as
// as 0, so the same struct can be used for every local-dynamic variable
// used in the shared-object.
// OutputRelocs:      Relocation section '.rela.dyn' at offset 0x{{[0-9a-f]+}} contains 1 entries:
// OutputRelocs-NEXT: Offset Info Type Symbol's Value Symbol's Name + Addend
// OutputRelocs-NEXT: R_PPC64_DTPMOD64

// Check that the got has 3 entries, 1 for the TOC and 1 stucture of 2 entries
// for the tls variables. Also verify the address so we can check the offsets
// we calculate for each relocation type.
// CheckGot: got          00000018 0000000000020100

// got starts at 0x20100 so .TOC. will be 0x28100, and the tls_index struct is
// at 0x20108.

// #ha(i@got@tlsld) --> (0x20108 - 0x28100 + 0x8000) >> 16 = 0
// #lo(i@got@tlsld) --> (0x20108 - 0x28100) = -7ff8 = -32760
// Dis:     test:
// Dis:        addis 3, 2, 0
// Dis-NEXT:   addi 3, 3, -32760

// #hi(j@got@tlsld) --> (0x20108 - 0x28100 ) > 16 = -1
// Dis: test_hi:
// Dis:   lis 3, -1

// k@got@tlsld --> (0x20108 - 0x28100) = -7ff8 = -32760
// Dis: test_16:
// Dis:   li 3, -32760
