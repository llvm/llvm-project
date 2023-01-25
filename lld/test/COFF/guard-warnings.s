# REQUIRES: x86
# Make a DLL that exports exportfn1.
# RUN: yaml2obj %p/Inputs/export.yaml -o %basename_t-exp.obj
# RUN: lld-link /out:%basename_t-exp.dll /dll %basename_t-exp.obj /export:exportfn1 /implib:%basename_t-exp.lib
# RUN: split-file %s %t
# RUN: llvm-mc -triple x86_64-windows-msvc %t/main.s -filetype=obj -o %t/main.obj

# RUN: lld-link %t/main.obj -guard:cf,longjmp,ehcont -out:%t-missing.exe -entry:main %basename_t-exp.lib 2>&1 | FileCheck %s --check-prefix=WARN_MISSING
# WARN_MISSING: warning: Control Flow Guard is enabled but '_load_config_used' is missing

# RUN: llvm-mc -triple x86_64-windows-msvc %t/loadcfg-invalid.s -filetype=obj -o %t/loadcfg-invalid.obj
# RUN: lld-link %t/main.obj %t/loadcfg-invalid.obj -guard:cf,longjmp,ehcont -out:%t-invalid.exe -entry:main %basename_t-exp.lib 2>&1 | FileCheck %s --check-prefix=WARN_INVALID
# WARN_INVALID:      warning: GuardCFFunctionTable not set correctly in '_load_config_used'
# WARN_INVALID-NEXT: warning: GuardCFFunctionCount not set correctly in '_load_config_used'
# WARN_INVALID-NEXT: warning: GuardFlags not set correctly in '_load_config_used'
# WARN_INVALID-NEXT: warning: GuardAddressTakenIatEntryTable not set correctly in '_load_config_used'
# WARN_INVALID-NEXT: warning: GuardAddressTakenIatEntryCount not set correctly in '_load_config_used'
# WARN_INVALID-NEXT: warning: GuardLongJumpTargetTable not set correctly in '_load_config_used'
# WARN_INVALID-NEXT: warning: GuardLongJumpTargetCount not set correctly in '_load_config_used'
# WARN_INVALID-NEXT: warning: GuardEHContinuationTable not set correctly in '_load_config_used'
# WARN_INVALID-NEXT: warning: GuardEHContinuationCount not set correctly in '_load_config_used'

# RUN: llvm-mc -triple x86_64-windows-msvc %t/loadcfg-small112.s -filetype=obj -o %t/loadcfg-small112.obj
# RUN: lld-link %t/main.obj %t/loadcfg-small112.obj -guard:cf,longjmp -out:%t-small112.exe -entry:main %basename_t-exp.lib 2>&1 | FileCheck %s --check-prefix=WARN_SMALL_112
# WARN_SMALL_112: warning: '_load_config_used' structure too small to include GuardFlags

# RUN: llvm-mc -triple x86_64-windows-msvc %t/loadcfg-small148.s -filetype=obj -o %t/loadcfg-small148.obj
# RUN: lld-link %t/main.obj %t/loadcfg-small148.obj -guard:cf,longjmp -out:%t-small148.exe -entry:main %basename_t-exp.lib 2>&1 | FileCheck %s --check-prefix=WARN_SMALL_148
# WARN_SMALL_148: warning: '_load_config_used' structure too small to include GuardLongJumpTargetCount

# RUN: llvm-mc -triple x86_64-windows-msvc %t/loadcfg-small244.s -filetype=obj -o %t/loadcfg-small244.obj
# RUN: lld-link %t/main.obj %t/loadcfg-small244.obj -guard:cf,longjmp,ehcont -out:%t-small244.exe -entry:main %basename_t-exp.lib 2>&1 | FileCheck %s --check-prefix=WARN_SMALL_244
# WARN_SMALL_244: warning: '_load_config_used' structure too small to include GuardEHContinuationCount

# RUN: llvm-mc -triple x86_64-windows-msvc %t/loadcfg-misaligned1.s -filetype=obj -o %t/loadcfg-misaligned1.obj
# RUN: lld-link %t/main.obj %t/loadcfg-misaligned1.obj -guard:cf,longjmp,ehcont -out:%t-misaligned1.exe -entry:main %basename_t-exp.lib 2>&1 | FileCheck %s --check-prefix=WARN_ALIGN1
# WARN_ALIGN1: warning: '_load_config_used' is misaligned (expected alignment to be 8 bytes, got 4 instead)

# RUN: llvm-mc -triple x86_64-windows-msvc %t/loadcfg-misaligned2.s -filetype=obj -o %t/loadcfg-misaligned2.obj
# RUN: lld-link %t/main.obj %t/loadcfg-misaligned2.obj -guard:cf,longjmp,ehcont -out:%t-misaligned2.exe -entry:main %basename_t-exp.lib 2>&1 | FileCheck %s --check-prefix=WARN_ALIGN2
# WARN_ALIGN2: warning: '_load_config_used' is misaligned (RVA is 0x{{[0-9A-F]*}}2 not aligned to 8 bytes)

# RUN: llvm-mc -triple x86_64-windows-msvc %t/loadcfg-full.s -filetype=obj -o %t/loadcfg-full.obj
# RUN: lld-link %t/main.obj %t/loadcfg-full.obj -guard:cf,longjmp,ehcont -out:%t.exe -entry:main %basename_t-exp.lib 2>&1 | FileCheck %s --check-prefix=NOWARN --allow-empty
# NOWARN-NOT: warning

# Sanity check to make sure the no-warn version has the expected data.
# RUN: llvm-readobj --file-headers --coff-load-config %t.exe | FileCheck --check-prefix=CHECK %s
# CHECK: ImageBase: 0x140000000
# CHECK: LoadConfig [
# CHECK:   SEHandlerTable: 0x0
# CHECK:   SEHandlerCount: 0
# CHECK:   GuardCFCheckFunction: 0x0
# CHECK:   GuardCFCheckDispatch: 0x0
# CHECK:   GuardCFFunctionTable: 0x14000{{([0-9A-F]{4})}}
# CHECK:   GuardCFFunctionCount: 1
# CHECK:   GuardFlags [ (0x410500)
# CHECK:     CF_FUNCTION_TABLE_PRESENT (0x400)
# CHECK:     CF_INSTRUMENTED (0x100)
# CHECK:     CF_LONGJUMP_TABLE_PRESENT (0x10000)
# CHECK:     EH_CONTINUATION_TABLE_PRESENT (0x400000)
# CHECK:   ]
# CHECK:   GuardAddressTakenIatEntryTable: 0x14000{{([0-9A-F]{4})}}
# CHECK:   GuardAddressTakenIatEntryCount: 1
# CHECK:   GuardLongJumpTargetTable: 0x14000{{([0-9A-F]{4})}}
# CHECK:   GuardLongJumpTargetCount: 1
# CHECK:   GuardEHContinuationTable: 0x14000{{([0-9A-F]{4})}}
# CHECK:   GuardEHContinuationCount: 1
# CHECK: ]
# CHECK-NEXT: GuardFidTable [
# CHECK-NEXT:   0x14000{{([0-9A-F]{4})}}
# CHECK-NEXT: ]
# CHECK-NEXT: GuardIatTable [
# CHECK-NEXT:   0x14000{{([0-9A-F]{4})}}
# CHECK-NEXT: ]
# CHECK-NEXT: GuardLJmpTable [
# CHECK-NEXT:   0x14000{{([0-9A-F]{4})}}
# CHECK-NEXT: ]
# CHECK-NEXT: GuardEHContTable [
# CHECK-NEXT:   0x14000{{([0-9A-F]{4})}}
# CHECK-NEXT: ]


#--- main.s

# We need @feat.00 to have 0x4000 | 0x800 to indicate /guard:cf and /guard:ehcont.
        .def     @feat.00;
        .scl    3;
        .type   0;
        .endef
        .globl  @feat.00
@feat.00 = 0x4800
        .def     main; .scl    2; .type   32; .endef
        .globl	main                            # -- Begin function main
        .p2align	4, 0x90
main:
        mov %eax, %eax
	movq __imp_exportfn1(%rip), %rax
        callq *%rax
        nop
# Fake setjmp target
$cfgsj_main0:
        mov %ebx, %ebx
        nop
# Fake ehcont target
$ehgcr_0_1:
        mov %ecx, %ecx
        nop
        retq
                                        # -- End function
        .section	.gfids$y,"dr"
        .symidx main
        .section	.giats$y,"dr"
        .symidx __imp_exportfn1
        .section	.gljmp$y,"dr"
        .symidx $cfgsj_main0
        .section	.gehcont$y,"dr"
        .symidx	$ehgcr_0_1
        .addrsig_sym main
        .addrsig_sym __imp_exportfn1
        .section  .rdata,"dr"

#--- loadcfg-invalid.s

.globl _load_config_used
        .p2align 3
_load_config_used:
        .long 312
        .fill 308, 1, 0

#--- loadcfg-small112.s

.globl _load_config_used
        .p2align 3
_load_config_used:
        .long 112
        .fill 108, 1, 0

#--- loadcfg-small148.s

.globl _load_config_used
        .p2align 3
_load_config_used:
        .long 148
        .fill 124, 1, 0
        .quad __guard_fids_table
        .quad __guard_fids_count
        .long __guard_flags

#--- loadcfg-small244.s

.globl _load_config_used
        .p2align 3
_load_config_used:
        .long 244
        .fill 124, 1, 0
        .quad __guard_fids_table
        .quad __guard_fids_count
        .long __guard_flags
        .fill 12, 1, 0
        .quad __guard_iat_table
        .quad __guard_iat_count
        .quad __guard_longjmp_table
        .quad __guard_longjmp_count
        .fill 52, 1, 0          # Up to HotPatchTableOffset

#--- loadcfg-misaligned1.s

.globl _load_config_used
        .fill 2, 1, 0           # offset by 2
        .p2align 2              # then align to 0x04
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

#--- loadcfg-misaligned2.s

.globl _load_config_used
        .p2align 4              # align to 0x10
        .fill 2, 1, 0           # then offset by 2
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

#--- loadcfg-full.s

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
