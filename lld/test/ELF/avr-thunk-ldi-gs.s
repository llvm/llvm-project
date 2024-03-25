; REQUIRES: avr
; RUN: llvm-mc -filetype=obj -triple=avr -mcpu=atmega2560 %s -o %t.o
; RUN: ld.lld %t.o --defsym=a=0x1fffe --defsym=b=0x20000 -o %t
; RUN: llvm-objdump -d --print-imm-hex --no-show-raw-insn --mcpu=atmega2560 %t \
; RUN:     | FileCheck %s

.section .LDI,"ax",@progbits

;; CHECK-LABEL: <__AVRThunk_b>:
;; CHECK-NEXT:  110b4: jmp  0x20000
;; CHECK-LABEL: <__init>:
;; CHECK-NEXT:  110b8:  ldi r30, 0xff
;; CHECK-NEXT:  110ba:  ldi r31, 0xff
;; CHECK-NEXT:  110bc:  eicall
;; The destination of the following two LDI instructions is
;; __AVRThunk_b == 0x110b4, so they actually are
;;                      ldi r30, ((0x110b4) >> 1) & 0xff
;;                      ldi r31, ((0x110b4) >> 9)
;; CHECK-NEXT:  110be:  ldi r30, 0x5a
;; CHECK-NEXT:  110c0:  ldi r31, 0x88
;; CHECK-NEXT:  110c2:  eicall
;; CHECK-NOT:   __AVRThunk_a

.globl __init
__init:
;; No thunk is needed, since the destination is in range [0, 0x1ffff].
ldi r30, lo8_gs(a)  ; R_AVR_LO8_LDI_GS
ldi r31, hi8_gs(a)  ; R_AVR_HI8_LDI_GS
eicall
;; A thunk is needed, since the destination is out of range.
ldi r30, lo8_gs(b)  ; R_AVR_LO8_LDI_GS
ldi r31, hi8_gs(b)  ; R_AVR_HI8_LDI_GS
eicall
