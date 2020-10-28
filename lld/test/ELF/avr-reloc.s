; REQUIRES: avr
; RUN: llvm-mc -filetype=obj -triple=avr -mcpu=atmega328p %s -o %t.o
; RUN: ld.lld %t.o --defsym=a=0x12345678 --defsym=b=30 -o %t
; RUN: llvm-objdump -d --print-imm-hex %t | FileCheck %s
; RUN: llvm-objdump -s %t | FileCheck --check-prefix=HEX %s

.section .LDI,"ax",@progbits
; CHECK-LABEL: section .LDI:
; CHECK:       ldi     r20, 0x78
; CHECK-NEXT:  ldi     r20, 0x56
; CHECK-NEXT:  ldi     r20, 0x34
; CHECK-NEXT:  ldi     r20, 0x12
; CHECK-NEXT:  ldi     r20, 0x3c
; CHECK-NEXT:  ldi     r20, 0x2b
; CHECK-NEXT:  ldi     r20, 0x1a
; CHECK-NEXT:  ldi     r20, 0xff
ldi r20, lo8(a)     ; R_AVR_LO8_LDI
ldi r20, hi8(a)     ; R_AVR_HI8_LDI
ldi r20, hh8(a)     ; R_AVR_HH8_LDI
ldi r20, hhi8(a)    ; R_AVR_MS8_LDI

ldi r20, pm_lo8(a)  ; R_AVR_LO8_LDI_PM
ldi r20, pm_hi8(a)  ; R_AVR_HI8_LDI_PM
ldi r20, pm_hh8(a)  ; R_AVR_HH8_LDI_PM

ldi r20, b+225

.section .LDI_NEG,"ax",@progbits
; CHECK-LABEL: section .LDI_NEG:
; CHECK:       ldi     r20, 0x88
; CHECK-NEXT:  ldi     r20, 0xa9
; CHECK-NEXT:  ldi     r20, 0xcb
; CHECK-NEXT:  ldi     r20, 0xed
; CHECK-NEXT:  ldi     r20, 0xc4
; CHECK-NEXT:  ldi     r20, 0xd4
; CHECK-NEXT:  ldi     r20, 0xe5
ldi r20, lo8(-(a))     ; R_AVR_LO8_LDI_NEG
ldi r20, hi8(-(a))     ; R_AVR_HI8_LDI_NEG
ldi r20, hh8(-(a))     ; R_AVR_HH8_LDI_NEG
ldi r20, hhi8(-(a))    ; R_AVR_MS8_LDI_NEG

ldi r20, pm_lo8(-(a))  ; R_AVR_LO8_LDI_PM_NEG
ldi r20, pm_hi8(-(a))  ; R_AVR_HI8_LDI_PM_NEG
ldi r20, pm_hh8(-(a))  ; R_AVR_HH8_LDI_PM_NEG

;; The disassembler is not yet able to decode those opcodes
;; 9e 8e    std    Y+30, r9
;; 9e 8c    ldd    r9, Y+30
;; 4e 96    adiw   r24, 0x1e
.section .SIX,"ax",@progbits
; HEX-LABEL: section .SIX:
; HEX-NEXT:  9e8e9e8c 4e96
std Y+b, r9   ; R_AVR_6
ldd r9, Y+b   ; R_AVR_6
adiw r24, b   ; R_AVR_6_ADIW

.section .PORT,"ax",@progbits
; CHECK-LABEL: section .PORT:
; CHECK:       in     r20, 0x1e
; CHECK-NEXT:  sbic   0x1e, 0x1
in    r20, b  ; R_AVR_PORT6
sbic  b, 1    ; R_AVR_PORT5

;; The disassembler is not yet able to decode those opcodes
;; 0f c0    rjmp   .+30
;; ee cf    rjmp   .-36
;; 69 f0    breq   .+26
;; 61 f3    breq   .-40
.section .PCREL,"ax",@progbits
; HEX-LABEL: section .PCREL:
; HEX-NEXT:  0fc0eecf 69f061f3
foo:
rjmp foo + 32  ; R_AVR_13_PCREL
rjmp foo - 32  ; R_AVR_13_PCREL
breq foo + 32  ; R_AVR_7_PCREL
breq foo - 32  ; R_AVR_7_PCREL

.section .DATA,"ax",@progbits
; HEX-LABEL: section .DATA:
; HEX-NEXT:  {{.*}} 1e1e000f 00785634 12
.byte b        ; R_AVR_8
.short b       ; R_AVR_16
.short gs(b)   ; R_AVR_16_PM
.long a        ; R_AVR_32
