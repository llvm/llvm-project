// This test checks whether BOLT can correctly handle relocations against weak symbols.

// RUN: %clang %cflags -Wl,-z,notext -shared -Wl,-q %s -o %t.so
// RUN: llvm-bolt %t.so -o %t.so.bolt
// RUN: llvm-nm -n %t.so.bolt > %t.out.txt
// RUN: llvm-objdump -z -dj .rodata %t.so.bolt >> %t.out.txt
// RUN: FileCheck %s --input-file=%t.out.txt

# CHECK: w func_1
# CHECK: {{0+}}[[#%x,ADDR:]] W func_2

# CHECK: {{.*}} <.rodata>:
# CHECK-NEXT: {{.*}} .word 0x00000000
# CHECK-NEXT: {{.*}} .word 0x00000000
# CHECK-NEXT: {{.*}} .word 0x{{[0]+}}[[#ADDR]]
# CHECK-NEXT: {{.*}} .word 0x00000000

	.text
	.weak	func_2
	.weak	func_1
	.global	wow
	.type	wow, %function
wow:
	bl func_1
	bl func_2
	ret
	.type   func_2, %function
func_2:
	ret
	.section	.rodata
.LC0:
	.xword	func_1
.LC1:
	.xword	func_2
