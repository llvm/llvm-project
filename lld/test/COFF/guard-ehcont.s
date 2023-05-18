# REQUIRES: x86
# RUN: llvm-mc -triple x86_64-windows-msvc %s -filetype=obj -o %t.obj
# RUN: lld-link %t.obj -guard:cf -guard:ehcont -out:%t.exe -entry:main
# RUN: llvm-readobj --file-headers --coff-load-config %t.exe | FileCheck %s

# CHECK: ImageBase: 0x140000000
# CHECK: LoadConfig [
# CHECK:   SEHandlerTable: 0x0
# CHECK:   SEHandlerCount: 0
# CHECK:   GuardCFCheckFunction: 0x0
# CHECK:   GuardCFCheckDispatch: 0x0
# CHECK:   GuardCFFunctionTable: 0x14000{{.*}}
# CHECK:   GuardCFFunctionCount: 1
# CHECK:   GuardFlags [ (0x400500)
# CHECK:     CF_FUNCTION_TABLE_PRESENT (0x400)
# CHECK:     CF_INSTRUMENTED (0x100)
# CHECK:     EH_CONTINUATION_TABLE_PRESENT (0x400000)
# CHECK:   ]
# CHECK:   GuardAddressTakenIatEntryTable: 0x0
# CHECK:   GuardAddressTakenIatEntryCount: 0
# CHECK:   GuardEHContinuationTable: 0x14000{{.*}}
# CHECK:   GuardEHContinuationCount: 2
# CHECK: ]
# CHECK:      GuardEHContTable [
# CHECK-NEXT:   0x14000{{.*}}
# CHECK-NEXT:   0x14000{{.*}}
# CHECK-NEXT: ]

# We need @feat.00 to have 0x4000 to indicate /guard:ehcont.
        .def     @feat.00;
        .scl    3;
        .type   0;
        .endef
        .globl  @feat.00
@feat.00 = 0x4000
        .def     main; .scl    2; .type   32; .endef
        .globl	main                            # -- Begin function main
        .p2align	4, 0x90
main:
.seh_proc main
        .seh_handler __C_specific_handler, @unwind, @except
        .seh_handlerdata
        .long 2
        .long (seh_begin)@IMGREL
        .long (seh_end)@IMGREL
        .long 1
        .long (seh_except)@IMGREL
        .long (seh2_begin)@IMGREL
        .long (seh2_end)@IMGREL
        .long 1
        .long (seh2_except)@IMGREL
        .text
    seh_begin:
        nop
        int3
        nop
    seh_end:
        nop
    seh_except:
        nop

    seh2_begin:
        nop
        int3
        nop
    seh2_end:
        nop
    seh2_except:
        nop

        xor %eax, %eax
        ret
.seh_endproc

__C_specific_handler:
        ret

.section	.gehcont$y,"dr"
.symidx	seh_except
.symidx	seh2_except

.section  .rdata,"dr"
.globl _load_config_used
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
