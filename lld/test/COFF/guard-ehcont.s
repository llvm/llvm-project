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
# CHECK:   GuardEHContinuationCount: 1
# CHECK: ]
# CHECK:      GuardEHContTable [
# CHECK-NEXT:   0x14000{{.*}}
# CHECK-NEXT: ]


# This assembly is reduced from C code like:
# int main()
# {
#   try {
#     throw 3;
#   }
#   catch (int e) {
#     return e != 3;
#   }
#   return 2;
# }

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
main:                                   # @main
.Lfunc_begin0:
.seh_proc main
.intel_syntax
        .seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:                                # %entry
        push	rbp
        .seh_pushreg rbp
        sub	rsp, 64
        .seh_stackalloc 64
        lea	rbp, [rsp + 64]
        .seh_setframe rbp, 64
        .seh_endprologue
        mov	qword ptr [rbp - 16], -2
        mov	dword ptr [rbp - 20], 0
        mov	dword ptr [rbp - 24], 3
.Ltmp0:
        lea	rdx, [rip + _TI1H]
        lea	rcx, [rbp - 24]
        call	_CxxThrowException
.Ltmp1:
        jmp	.LBB0_3
.LBB0_2:                                # Block address taken
                                        # %catchret.dest
$ehgcr_0_2:
        mov	eax, dword ptr [rbp - 20]
        add	rsp, 64
        pop	rbp
        ret
.LBB0_3:                                # %unreachable
        int3
        .seh_handlerdata
        .long	($cppxdata$main)@IMGREL
        .text
        .seh_endproc
        .def	 "?catch$1@?0?main@4HA";
        .scl	3;
        .type	32;
        .endef
        .p2align	4, 0x90
"?catch$1@?0?main@4HA":
.seh_proc "?catch$1@?0?main@4HA"
        .seh_handler __CxxFrameHandler3, @unwind, @except
.LBB0_1:                                # %catch
        mov	qword ptr [rsp + 16], rdx
        push	rbp
        .seh_pushreg rbp
        sub	rsp, 32
        .seh_stackalloc 32
        lea	rbp, [rdx + 64]
        .seh_endprologue
        mov	eax, dword ptr [rbp - 4]
        sub	eax, 3
        setne	al
        movzx	eax, al
        mov	dword ptr [rbp - 20], eax
        lea	rax, [rip + .LBB0_2]
        add	rsp, 32
        pop	rbp
        ret                                     # CATCHRET
        .def     free; .scl    2; .type   32; .endef
        .globl  free
free:
        ret
        .def     __CxxFrameHandler3; .scl    2; .type   32; .endef
        .globl  __CxxFrameHandler3
__CxxFrameHandler3:
        ret
        .def     _CxxThrowException; .scl    2; .type   32; .endef
        .globl  _CxxThrowException
_CxxThrowException:
        ret
.Lfunc_end0:
        .seh_handlerdata
        .long	($cppxdata$main)@IMGREL
        .text
        .seh_endproc
        .section	.xdata,"dr"
        .p2align	2
$cppxdata$main:
        .long	429065506                       # MagicNumber
        .long	2                               # MaxState
        .long	($stateUnwindMap$main)@IMGREL   # UnwindMap
        .long	1                               # NumTryBlocks
        .long	($tryMap$main)@IMGREL           # TryBlockMap
        .long	4                               # IPMapEntries
        .long	($ip2state$main)@IMGREL         # IPToStateXData
        .long	48                              # UnwindHelp
        .long	0                               # ESTypeList
        .long	1                               # EHFlags
$stateUnwindMap$main:
        .long	-1                              # ToState
        .long	0                               # Action
        .long	-1                              # ToState
        .long	0                               # Action
$tryMap$main:
        .long	0                               # TryLow
        .long	0                               # TryHigh
        .long	1                               # CatchHigh
        .long	1                               # NumCatches
        .long	($handlerMap$0$main)@IMGREL     # HandlerArray
$handlerMap$0$main:
        .long	0                               # Adjectives
        .long	"??_R0H@8"@IMGREL               # Type
        .long	60                              # CatchObjOffset
        .long	"?catch$1@?0?main@4HA"@IMGREL   # Handler
        .long	56                              # ParentFrameOffset
$ip2state$main:
        .long	.Lfunc_begin0@IMGREL            # IP
        .long	-1                              # ToState
        .long	.Ltmp0@IMGREL+1                 # IP
        .long	0                               # ToState
        .long	.Ltmp1@IMGREL+1                 # IP
        .long	-1                              # ToState
        .long	"?catch$1@?0?main@4HA"@IMGREL   # IP
        .long	1                               # ToState
        .text
                                        # -- End function
        .section	.data,"dw",discard,"??_R0H@8"
        .globl	"??_R0H@8"                      # @"??_R0H@8"
        .p2align	4
"??_R0H@8":
        .quad	0
        .quad	0
        .asciz	".H"
        .zero	5

        .section	.xdata,"dr",discard,"_CT??_R0H@84"
        .globl	"_CT??_R0H@84"                  # @"_CT??_R0H@84"
        .p2align	4
"_CT??_R0H@84":
        .long	1                               # 0x1
        .long	"??_R0H@8"@IMGREL
        .long	0                               # 0x0
        .long	4294967295                      # 0xffffffff
        .long	0                               # 0x0
        .long	4                               # 0x4
        .long	0                               # 0x0

        .section	.xdata,"dr",discard,_CTA1H
        .globl	_CTA1H                          # @_CTA1H
        .p2align	3
_CTA1H:
        .long	1                               # 0x1
        .long	"_CT??_R0H@84"@IMGREL

        .section	.xdata,"dr",discard,_TI1H
        .globl	_TI1H                           # @_TI1H
        .p2align	3
_TI1H:
        .long	0                               # 0x0
        .long	0                               # 0x0
        .long	0                               # 0x0
        .long	_CTA1H@IMGREL

        .section	.gehcont$y,"dr"
        .symidx	$ehgcr_0_2
        .addrsig_sym _CxxThrowException
        .addrsig_sym __CxxFrameHandler3
        .addrsig_sym "??_R0H@8"
        .addrsig_sym __ImageBase
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