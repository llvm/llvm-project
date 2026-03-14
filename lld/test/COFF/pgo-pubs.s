# REQUIRES: x86
# RUN: llvm-mc -triple=x86_64-windows-msvc -filetype=obj -o %t.obj %s
# RUN: lld-link %t.obj -out:%t.exe -pdb:%t.pdb -debug -entry:main -subsystem:console
# RUN: llvm-pdbutil dump -publics %t.pdb | FileCheck %s

# Check that there are no __prof[dc] or __covrec public symbols.

# CHECK-NOT: __profd
# CHECK-NOT: __profc
# CHECK-NOT: __covrec
# CHECK: S_PUB32 {{.*}} `main`
# CHECK-NOT: __profd
# CHECK-NOT: __profc
# CHECK-NOT: __covrec


# The following assembly is simplified from this C code:
# int main() {
#   return 0;
# }

# Compiled like so:
# clang-cl -c pgo-pubs.c -fprofile-instr-generate -fcoverage-mapping -clang:-save-temps


	.text
	.intel_syntax noprefix
	.globl	main                            # -- Begin function main
main:                                   # @main
# %bb.0:                                # %entry
	xor	eax, eax
	ret

	.section	.lcovfun$M,"dr",discard,__covrec_DB956436E78DD5FAu
	.globl	__covrec_DB956436E78DD5FAu      # @__covrec_DB956436E78DD5FAu
	.p2align	3
__covrec_DB956436E78DD5FAu:
	.quad	-2624081020897602054            # 0xdb956436e78dd5fa
	.long	9                               # 0x9
	.quad	24                              # 0x18
	.quad	2164039332547799183             # 0x1e08364eb07c288f
	.ascii	"\001\001\000\001\001\b\f\002\002"

	.section	.lcovmap$M,"dr"
	.p2align	3                               # @__llvm_coverage_mapping
.L__llvm_coverage_mapping:
	.long	0                               # 0x0
	.long	40                              # 0x28
	.long	0                               # 0x0
	.long	5                               # 0x5
	.ascii	"\002%\000\031C:\\src\\llvm-project\\build\npgo-pubs.i"

	.section	.lprfc$M,"dw"
	.p2align	3                               # @__profc_main
__profc_main:
	.zero	8

	.section	.lprfd$M,"dw"
	.p2align	3                               # @__profd_main
__profd_main:
	.quad	-2624081020897602054            # 0xdb956436e78dd5fa
	.quad	24                              # 0x18
	.quad	__profc_main
	.quad	main
	.quad	0
	.long	1                               # 0x1
	.zero	4

	.section	.lprfn$M,"dr"
.L__llvm_prf_nm:                        # @__llvm_prf_nm
	.ascii	"\004\000main"
