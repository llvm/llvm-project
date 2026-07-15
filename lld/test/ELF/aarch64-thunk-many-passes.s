// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t
// RUN: ld.lld %t -o %t1 2>&1 | FileCheck %s

// CHECK-NOT: error: address assignment did not converge

.section .text.call1, "ax", %progbits
.balign 8
.global _start
_start:
  bl fn1
  .space 16
.section .text.call2, "ax", %progbits
  bl fn2
  .space 16
.section .text.call3, "ax", %progbits
  bl fn3
  .space 16
.section .text.call4, "ax", %progbits
  bl fn4
  .space 16
.section .text.call5, "ax", %progbits
  bl fn5
  .space 16
.section .text.call6, "ax", %progbits
  bl fn6
  .space 16
.section .text.call7, "ax", %progbits
  bl fn7
  .space 16
.section .text.call8, "ax", %progbits
  bl fn8
  .space 16
.section .text.call9, "ax", %progbits
  bl fn9
  .space 16
.section .text.call10, "ax", %progbits
  bl fn10
  .space 16
.section .text.call11, "ax", %progbits
  bl fn11
  .space 16
.section .text.call12, "ax", %progbits
  bl fn12
  .space 16
.section .text.call13, "ax", %progbits
  bl fn13
  .space 16
.section .text.call14, "ax", %progbits
  bl fn14
  .space 16
.section .text.call15, "ax", %progbits
  bl fn15
  .space 16
.section .text.call16, "ax", %progbits
  bl fn16
  .space 16
.section .text.call17, "ax", %progbits
  bl fn17
  .space 16
.section .text.call18, "ax", %progbits
  bl fn18
  .space 16
.section .text.call19, "ax", %progbits
  bl fn19
  .space 16
.section .text.call20, "ax", %progbits
  bl fn20
  .space 16
.section .text.call21, "ax", %progbits
  bl fn21
  .space 16
.section .text.call22, "ax", %progbits
  bl fn22
  .space 16
.section .text.call23, "ax", %progbits
  bl fn23
  .space 16
.section .text.call24, "ax", %progbits
  bl fn24
  .space 16
.section .text.call25, "ax", %progbits
  bl fn25
  .space 16
.section .text.call26, "ax", %progbits
  bl fn26
  .space 16
.section .text.call27, "ax", %progbits
  bl fn27
  .space 16
.section .text.call28, "ax", %progbits
  bl fn28
  .space 16
.section .text.call29, "ax", %progbits
  bl fn29
  .space 16
.section .text.call30, "ax", %progbits
  bl fn30
  .space 16
.section .text.call31, "ax", %progbits
  bl fn31
  .space 16
.section .text.call32, "ax", %progbits
  bl fn32

.section .text.space, "ax", %progbits
.space 134217232

.section .text.targets, "ax", %progbits
.balign 4
.global fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8, fn9, fn10, fn11, fn12, fn13, fn14, fn15, fn16, fn17, fn18, fn19, fn20, fn21, fn22, fn23, fn24, fn25, fn26, fn27, fn28, fn29, fn30, fn31, fn32
fn1:
  ret
.space 12
fn2:
  ret
.space 12
fn3:
  ret
.space 12
fn4:
  ret
.space 12
fn5:
  ret
.space 12
fn6:
  ret
.space 12
fn7:
  ret
.space 12
fn8:
  ret
.space 12
fn9:
  ret
.space 12
fn10:
  ret
.space 12
fn11:
  ret
.space 12
fn12:
  ret
.space 12
fn13:
  ret
.space 12
fn14:
  ret
.space 12
fn15:
  ret
.space 12
fn16:
  ret
.space 12
fn17:
  ret
.space 12
fn18:
  ret
.space 12
fn19:
  ret
.space 12
fn20:
  ret
.space 12
fn21:
  ret
.space 12
fn22:
  ret
.space 12
fn23:
  ret
.space 12
fn24:
  ret
.space 12
fn25:
  ret
.space 12
fn26:
  ret
.space 12
fn27:
  ret
.space 12
fn28:
  ret
.space 12
fn29:
  ret
.space 12
fn30:
  ret
.space 12
fn31:
  ret
.space 12
fn32:
  ret
