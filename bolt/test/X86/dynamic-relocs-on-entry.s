// This test examines whether BOLT can correctly process when
// dynamic relocation points to other entry points of the
// function.

# RUN: %clang %cflags -fPIC -pie %s -o %t.exe -nostdlib -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt > %t.out.txt
# RUN: readelf -r %t.bolt >> %t.out.txt
# RUN: llvm-objdump --disassemble-symbols=chain %t.bolt >> %t.out.txt
# RUN: FileCheck %s --input-file=%t.out.txt

## Check if the new address in `chain` is correctly updated by BOLT
# CHECK: Relocation section '.rela.dyn' at offset 0x{{.*}} contains 1 entry:
# CHECK: {{.*}} R_X86_64_RELATIVE [[#%x,ADDR:]]
# CHECK: [[#ADDR]]: c3 retq
	.text
	.type   chain, @function
chain:
	movq    $1, %rax
Label:
	ret
	.size   chain, .-chain

	.type   _start, @function
	.global _start
_start:
	jmpq    *.Lfoo(%rip)
	ret
	.size   _start, .-_start

	.data
.Lfoo:
	.quad Label