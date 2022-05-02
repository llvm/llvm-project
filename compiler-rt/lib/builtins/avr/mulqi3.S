//===------------ mulhi3.S - int8 multiplication --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The corresponding C code is something like:
//
// int __mulqi3(char A, char B) {
//   return __mulhi3((int) A, (int) B);
// }
//
//===----------------------------------------------------------------------===//

	.text
	.align 2

	.globl __mulqi3
	.type  __mulqi3, @function

__mulqi3:
	mov    r25, r24
	lsl    r25
	sbc    r25, r25         ; Promote A from char to int: `(int) A`.
	mov    r23, r22
	lsl    r23
	sbc    r23, r23         ; Promote B from char to int: `(int) B`.
	rcall  __mulhi3         ; `__mulhi3((int) A, (int) B);`.
	ret
