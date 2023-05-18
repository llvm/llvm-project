# REQUIRES: x86

# The __guard_fids_table is a DefinedSynthetic when control flow guard is
# enabled and there are entries to be added to the fids table. This test uses
# this to check that DefinedSynthetic symbols are being written to the COFF
# symbol table.

# RUN: llvm-mc -triple x86_64-windows-msvc %s -filetype=obj -o %t.obj
# RUN: lld-link %t.obj -guard:cf -out:%t.exe -entry:main -debug:symtab
# RUN: llvm-readobj --symbols %t.exe | FileCheck --check-prefix=CHECK %s

# CHECK:      Name: __guard_fids_table
# CHECK-NEXT: Value:
# CHECK-NEXT: Section: .rdata (2)


# We need @feat.00 to have 0x800 to indicate /guard:cf.
        .def     @feat.00;
        .scl    3;
        .type   0;
        .endef
        .globl  @feat.00
@feat.00 = 0x800
        .def     main; .scl    2; .type   32; .endef
        .globl	main                            # -- Begin function main
        .p2align	4, 0x90
main:
        retq
                                        # -- End function
        .section	.gfids$y,"dr"
        .symidx main
        .section	.giats$y,"dr"
        .section	.gljmp$y,"dr"
        .addrsig_sym main
        .section  .rdata,"dr"

.globl _load_config_used
        .p2align 3
_load_config_used:
        .long 312
        .fill 124, 1, 0
        .quad __guard_fids_table
        .quad __guard_fids_count
        .long __guard_flags
        .fill 12, 1, 0
        .quad __guard_iat_table
        .quad __guard_iat_count
        .quad __guard_longjmp_table
        .quad __guard_longjmp_count
        .fill 72, 1, 0
        .quad __guard_eh_cont_table
        .quad __guard_eh_cont_count
        .fill 32, 1, 0
