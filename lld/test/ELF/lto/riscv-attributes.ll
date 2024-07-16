; REQUIRES: riscv

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llvm-mc -filetype=obj -triple=riscv32 1.s -o 1.o
; RUN: llvm-mc -filetype=obj -triple=riscv32 2.s -o 2.o
; RUN: llvm-as a.ll -o a.bc
; RUN: ld.lld 1.o 2.o a.bc -o out
; RUN: llvm-readelf --arch-specific out | FileCheck %s

; CHECK:      BuildAttributes {
; CHECK-NEXT:   FormatVersion: 0x41
; CHECK-NEXT:   Section 1 {
; CHECK-NEXT:     SectionLength: 79
; CHECK-NEXT:     Vendor: riscv
; CHECK-NEXT:     Tag: Tag_File (0x1)
; CHECK-NEXT:     Size: 69
; CHECK-NEXT:     FileAttributes {
; CHECK-NEXT:       Attribute {
; CHECK-NEXT:         Tag: 4
; CHECK-NEXT:         Value: 16
; CHECK-NEXT:         TagName: stack_align
; CHECK-NEXT:         Description: Stack alignment is 16-bytes
; CHECK-NEXT:       }
; CHECK-NEXT:       Attribute {
; CHECK-NEXT:         Tag: 6
; CHECK-NEXT:         Value: 1
; CHECK-NEXT:         TagName: unaligned_access
; CHECK-NEXT:         Description: Unaligned access
; CHECK-NEXT:       }
; CHECK-NEXT:       Attribute {
; CHECK-NEXT:         Tag: 5
; CHECK-NEXT:         TagName: arch
; CHECK-NEXT:         Value: rv32i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zmmul1p0_zbb1p0{{$}}
; CHECK-NEXT:       }
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT: }

;--- 1.s
.attribute 4, 16
.attribute 5, "rv32i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zmmul1p0"
;--- 2.s
.attribute 4, 16
.attribute 5, "rv32i2p1_m2p0_f2p2_d2p2_zbb1p0_zmmul1p0"
.attribute 6, 1

;--- a.ll
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32"

define void @_start() {
  ret void
}
