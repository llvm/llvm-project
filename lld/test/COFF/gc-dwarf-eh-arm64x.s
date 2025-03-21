# REQUIRES: aarch64

# RUN: llvm-mc -triple=aarch64-windows-gnu %s -filetype=obj -o %t-arm64.obj
# RUN: llvm-mc -triple=arm64ec-windows-gnu %s -filetype=obj -o %t-arm64ec.obj
# RUN: lld-link -machine:arm64x -lldmingw -out:%t.exe -opt:ref -entry:main %t-arm64.obj %t-arm64ec.obj -verbose 2>&1 | FileCheck %s
# CHECK: Discarded unused

# Check that __gxx_personality_v0 is not discarded and is present in the output.

# RUN: llvm-objdump -d %t.exe | FileCheck --check-prefix=DISASM %s
# DISASM:      0000000140001000 <.text>:
# DISASM-NEXT: 140001000: 52800000     mov     w0, #0x0                // =0
# DISASM-NEXT: 140001004: d65f03c0     ret
# DISASM-NEXT: 140001008: 52800020     mov     w0, #0x1                // =1
# DISASM-NEXT: 14000100c: d65f03c0     ret
# DISASM-NEXT:                 ...
# DISASM-NEXT: 140002000: 52800000     mov     w0, #0x0                // =0
# DISASM-NEXT: 140002004: d65f03c0     ret
# DISASM-NEXT: 140002008: 52800020     mov     w0, #0x1                // =1
# DISASM-NEXT: 14000200c: d65f03c0     ret

	.def main; .scl 2; .type 32; .endef
	.section .text,"xr",one_only,main
	.globl	main
main:
	.cfi_startproc
	.cfi_personality 0, __gxx_personality_v0
        mov w0, #0
	ret
	.cfi_endproc

	.def __gxx_personality_v0; .scl 2; .type 32; .endef
	.section .text,"xr",one_only,__gxx_personality_v0
	.globl	__gxx_personality_v0
__gxx_personality_v0:
        mov w0, #1
	ret

        .def unused; .scl 2; .type 32; .endef
	.section .text,"xr",one_only,unused
	.globl	unused
unused:
	.cfi_startproc
	.cfi_personality 0, __gxx_personality_v0
        mov w0, #2
	ret
	.cfi_endproc
