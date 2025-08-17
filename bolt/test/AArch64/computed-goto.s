// This test checks that BOLT creates entry points for addresses
// referenced by dynamic relocations.
// The test also checks that BOLT can map addresses inside functions.

// Checks for error and entry points.
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck %s
# RUN: llvm-bolt %t.exe -o %t.bolt --print-cfg | FileCheck --check-prefix=CHECK-ENTRIES %s

// Checks for dynamic relocations.
# RUN: llvm-readelf -dr %t.bolt > %t.out.txt
# RUN: llvm-objdump -j .rela.dyn -d %t.bolt >> %t.out.txt
# RUN: FileCheck --check-prefix=CHECK-RELOCS %s --input-file=%t.out.txt

// Before bolt could handle mapping addresses within moved functions, it
// would bail out with an error of the form:
// BOLT-ERROR: unable to get new address corresponding to input address 0x10390 in function main. Consider adding this function to --skip-funcs=...
// These addresses arise if computed GOTO is in use.
// Check that bolt does not emit any error.
# CHECK-NOT: BOLT-ERROR

// Check that there are dynamic relocations.
# CHECK-RELOCS:     Dynamic section at offset {{.*}} contains {{.*}} entries:
# CHECK-RELOCS:     Relocation section '.rela.dyn' at offset {{.*}} contains {{.*}} entries

// Check that dynamic relocations were updated
# CHECK-RELOCS: [[#%x,OFF:]] [[#%x,INFO_DYN:]] R_AARCH64_RELATIVE [[#%x,ADDR:]]
# CHECK-RELOCS-NEXT: [[#OFF + 8]] {{0*}}[[#INFO_DYN]] R_AARCH64_RELATIVE [[#ADDR + 8]]
# CHECK-RELOCS: [[#ADDR]] <unknown>
# CHECK-RELOCS: [[#ADDR + 8]] <unknown>

// Check that BOLT registers extra entry points for dynamic relocations.
# CHECK-ENTRIES: Binary Function "main" after building cfg {
# CHECK-ENTRIES:  IsMultiEntry: 1
# CHECK-ENTRIES: .Ltmp0 {{.*}}
# CHECK-ENTRIES-NEXT: Secondary Entry Point: {{.*}}
# CHECK-ENTRIES: .Ltmp1 {{.*}}
# CHECK-ENTRIES-NEXT: Secondary Entry Point: {{.*}}

.globl  main
.p2align        2
.type   main,@function
main:
.cfi_startproc
        adrp    x8, .L__const.main.ptrs+8
        add     x8, x8, :lo12:.L__const.main.ptrs+8
        ldr     x9, [x8], #8
        br      x9

.Label0: // Block address taken
        ldr     x9, [x8], #8
        br      x9

.Label1: // Block address taken
        mov     w0, #42
        ret

.Lfunc_end0:
.size   main, .Lfunc_end0-main
.cfi_endproc
        .type   .L__const.main.ptrs,@object
        .section        .data.rel.ro,"aw",@progbits
        .p2align        3, 0x0
.L__const.main.ptrs:
        .xword  .Label0
        .xword  .Label1
