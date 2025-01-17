## This test checks that indirect tail call is properly identified by BOLT on aarch64.
# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -O0 %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt --print-all --print-only=indirect  \
# RUN: %t.exe -o %t.bolt | FileCheck %s

#CHECK: Binary Function "indirect" after building cfg {
#CHECK-NOT: # UNKNOWN CONTROL FLOW
#CHECK: End of Function "indirect"
	.text
	.globl	indirect                       
	.type	indirect,@function
indirect:      
    cbz	x0, .LBB0_2                         
	ldr	x8, [x0]
	ldr	x1, [x8]
	br	x1
.LBB0_2:
	mov	w0, #3
	ret
	.size	indirect, .-indirect


	.globl	main                            
	.type	main,@function
main:                                   
	mov	w0, wzr
	ret
	.size	main, .-main
