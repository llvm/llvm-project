# REQUIRES: x86

# Make a DLL that exports exportfn1.
# RUN: yaml2obj %p/Inputs/export.yaml -o %basename_t-exp.obj
# RUN: lld-link /out:%basename_t-exp.dll /dll %basename_t-exp.obj /export:exportfn1 /implib:%basename_t-exp.lib

# Make an object file that imports exportfn1.
# RUN: llvm-mc -triple x86_64-windows-msvc %s -filetype=obj -o %basename_t.obj

# Check that the Guard address-taken IAT entry tables are propagated to the final executable.
# RUN: lld-link %basename_t.obj -guard:cf -guard:longjmp -entry:main -out:%basename_t-nodelay.exe %basename_t-exp.lib
# RUN: llvm-readobj --file-headers --coff-load-config %basename_t-nodelay.exe | FileCheck %s --check-prefix CHECK

# CHECK: ImageBase: 0x140000000
# CHECK: LoadConfig [
# CHECK:   GuardCFFunctionTable: 0x140002114
# CHECK:   GuardCFFunctionCount: 1
# CHECK:   GuardFlags [ (0x10500)
# CHECK:     CF_FUNCTION_TABLE_PRESENT (0x400)
# CHECK:     CF_INSTRUMENTED (0x100)
# CHECK:     CF_LONGJUMP_TABLE_PRESENT (0x10000)
# CHECK:   ]
# CHECK:   GuardAddressTakenIatEntryTable: 0x140002118
# CHECK:   GuardAddressTakenIatEntryCount: 1
# CHECK: ]
# CHECK:      GuardFidTable [
# CHECK-NEXT:   0x14000{{.*}}
# CHECK-NEXT: ]
# CHECK:      GuardIatTable [
# CHECK-NEXT:   0x14000{{.*}}
# CHECK-NEXT: ]


# Check that the additional load thunk symbol is added to the GFIDs table.
# RUN: lld-link %basename_t.obj -guard:cf -guard:longjmp -entry:main -out:%basename_t-delay.exe %basename_t-exp.lib -alternatename:__delayLoadHelper2=main -delayload:%basename_t-exp.dll
# RUN: llvm-readobj --file-headers --coff-load-config %basename_t-delay.exe | FileCheck %s --check-prefix DELAY-CHECK

# DELAY-CHECK: ImageBase: 0x140000000
# DELAY-CHECK: LoadConfig [
# DELAY-CHECK:   GuardCFFunctionTable: 0x140002114
# DELAY-CHECK:   GuardCFFunctionCount: 2
# DELAY-CHECK:   GuardFlags [ (0x10500)
# DELAY-CHECK:     CF_FUNCTION_TABLE_PRESENT (0x400)
# DELAY-CHECK:     CF_INSTRUMENTED (0x100)
# DELAY-CHECK:     CF_LONGJUMP_TABLE_PRESENT (0x10000)
# DELAY-CHECK:   ]
# DELAY-CHECK:   GuardAddressTakenIatEntryTable: 0x14000211C
# DELAY-CHECK:   GuardAddressTakenIatEntryCount: 1
# DELAY-CHECK: ]
# DELAY-CHECK:      GuardFidTable [
# DELAY-CHECK-NEXT:   0x14000{{.*}}
# DELAY-CHECK-NEXT:   0x14000{{.*}}
# DELAY-CHECK-NEXT: ]
# DELAY-CHECK:      GuardIatTable [
# DELAY-CHECK-NEXT:   0x14000{{.*}}
# DELAY-CHECK-NEXT: ]


# This assembly is reduced from C code like:
# __declspec(noinline)
# void IndirectCall(BOOL (func)(HANDLE)) {
#   (*func)(NULL);
# }
# int main(int argc, char** argv) {
#   IndirectCall(exportfn1);
# }

	.text
	.def	 @feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 2048
	.def	 IndirectCall;	.scl	2;	.type	32;	.endef
	.globl	IndirectCall                    # -- Begin function IndirectCall
	.p2align	4, 0x90
IndirectCall:                           # @IndirectCall
# %bb.0:
	subq	$40, %rsp
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rax
	movq	%rax, %rdx        # This would otherwise be: movq __guard_dispatch_icall_fptr(%rip), %rdx
	xorl	%ecx, %ecx
	callq	*%rdx
	nop
	addq	$40, %rsp
	retq
                                        # -- End function
	.def	 main;	.scl	2;	.type	32;	.endef
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
main:                                   # @main
# %bb.0:
	subq	$56, %rsp
	movq	__imp_exportfn1(%rip), %rax
	movq	%rdx, 48(%rsp)
	movl	%ecx, 44(%rsp)
	movq	%rax, %rcx
	callq	IndirectCall
	xorl	%eax, %eax
	addq	$56, %rsp
	retq
                                        # -- End function
	.section	.gfids$y,"dr"
	.section	.giats$y,"dr"
	.symidx	__imp_exportfn1
	.section	.gljmp$y,"dr"

# Load configuration directory entry (winnt.h _IMAGE_LOAD_CONFIG_DIRECTORY64).
# The linker will define the __guard_* symbols.
        .section .rdata,"dr"
.globl _load_config_used
_load_config_used:
        .long 256
        .fill 124, 1, 0
        .quad __guard_fids_table
        .quad __guard_fids_count
        .long __guard_flags
        .fill 12, 1, 0
        .quad __guard_iat_table
        .quad __guard_iat_count
        .quad __guard_longjmp_table
        .quad __guard_fids_count
        .fill 84, 1, 0