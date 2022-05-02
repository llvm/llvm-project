//===------------ mulhi3.S - int16 multiplication -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The corresponding C code is something like:
//
// int __mulhi3(int A, int B) {
//   int S = 0;
//   while (A != 0) {
//     if (A & 1)
//       S += B;
//     A = ((unsigned int) A) >> 1;
//     B <<= 1;
//   }
//   return S;
// }
//
//===----------------------------------------------------------------------===//

	.text
	.align 2

	.globl __mulhi3
	.type  __mulhi3, @function

__mulhi3:
	eor    r28, r28
	eor    r20, r20
	eor    r21, r21         ; Initialize the result to 0: `S = 0;`.

__mulhi3_loop:
	cp     r24, r28
	cpc    r25, r28         ; `while (A != 0) { ... }`
	breq   __mulhi3_end     ; End the loop if A is 0.

	mov    r29, r24
	andi   r29, 1           ; `if (A & 1) { ... }`
	breq   __mulhi3_loop_a  ; Omit the accumulation (`S += B;`) if  A's LSB is 0.

	add    r20, r22
	adc    r21, r23         ; Do the accumulation: `S += B;`.

__mulhi3_loop_a:
	lsr    r25
	ror    r24              ; `A = ((unsigned int) A) >> 1;`.
	lsl    r22
	rol    r23              ; `B <<= 1;`

	rjmp   __mulhi3_loop

__mulhi3_end:
	mov    r24, r20
	mov    r25, r21
	ret
