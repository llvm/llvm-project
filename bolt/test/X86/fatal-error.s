## Tests whether llvm-bolt will correctly exit with error code and printing
## fatal error message in case one occurs. Here we test opening a function
## reordering file that does not exist.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: not llvm-bolt %t.exe -o %t.null \
# RUN:   --reorder-blocks=normal --reorder-functions=user \
# RUN:   --function-order=/DOES/NOT/EXIST  2>&1 \
# RUN:   | FileCheck --check-prefix=CHECK %s

# CHECK: FATAL BOLT-ERROR: Ordered functions file "/DOES/NOT/EXIST" can't be opened

# Sample function reordering input, based off function-order-lite.s
  .globl main
  .type main, %function
main:
	.cfi_startproc
.LBB06:
	callq	func_a
	retq
	.cfi_endproc
.size main, .-main

  .globl func_a
  .type func_a, %function
func_a:
	.cfi_startproc
	retq
	.cfi_endproc
.size func_a, .-func_a

  .globl func_b
  .type func_b, %function
func_b:
	.cfi_startproc
	retq
	.cfi_endproc
.size func_b, .-func_b
