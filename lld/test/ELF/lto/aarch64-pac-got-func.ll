; REQUIRES: aarch64

; RUN: llvm-as %s -o %t.o
; RUN: ld.lld %t.o -shared -o %t
; RUN: llvm-readelf -r -x.got %t | FileCheck %s

; CHECK:      Relocation section '.rela.dyn' at offset 0x3d0 contains 8 entries:
; CHECK-NEXT:     Offset                                    Info             Type               Symbol's Value        Symbol's Name + Addend
; CHECK-NEXT: {{(0{8})}}[[OFFSET1:([[:xdigit:]]{8})]]  0000000100000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000       func_undef + 0
; CHECK-NEXT: {{([[:xdigit:]]{16})}}                   0000000200000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000       g1 + 0
; CHECK-NEXT: {{(0{8})}}[[OFFSET2:([[:xdigit:]]{8})]]  0000000300000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000       g2 + 0
; CHECK-NEXT: {{([[:xdigit:]]{16})}}                   0000000400000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000       g3 + 0
; CHECK-NEXT: {{(0{8})}}[[OFFSET3:([[:xdigit:]]{8})]]  0000000500000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000       g4 + 0
; CHECK-NEXT: {{([[:xdigit:]]{16})}}                   0000000600000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000       var_undef + 0
; CHECK-NEXT: {{(0{8})}}[[OFFSET4:([[:xdigit:]]{8})]]  0000000700000412 R_AARCH64_AUTH_GLOB_DAT {{([[:xdigit:]]{16})}} func + 0
; CHECK-NEXT: {{([[:xdigit:]]{16})}}                   0000000a00000412 R_AARCH64_AUTH_GLOB_DAT {{([[:xdigit:]]{16})}} var + 0

; CHECK:      Hex dump of section '.got':
; CHECK-NEXT: 0x[[OFFSET4]] 00000000 00000080 00000000 000000a0
;;                                         ^^ func: 0b10000000 bit 63 address diversity = true, bits 61..60 key = IA
;;                                                           ^^ var: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA
; CHECK-NEXT: 0x[[OFFSET1]] 00000000 00000080 00000000 000000a0
;;                                         ^^ func_undef: 0b10000000 bit 63 address diversity = true, bits 61..60 key = IA
;;                                                           ^^ g1: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA
; CHECK-NEXT: 0x[[OFFSET2]] 00000000 000000a0 00000000 000000a0
;;                                         ^^ g2: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA
;;                                                           ^^ g3: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA
; CHECK-NEXT: 0x[[OFFSET3]] 00000000 000000a0 00000000 000000a0
;;                                         ^^ g4: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA
;;                                                           ^^ var_undef: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

@g1 = external global ptr
@g2 = external global ptr
@g3 = external global ptr
@g4 = external global ptr

define void @func() {
entry:
  ret void
}
declare void @func_undef()

@var = global i32 42
@var_undef = external global i32

define void @bar() #0 {
entry:
  store ptr ptrauth (ptr @func, i32 0), ptr @g1
  store ptr ptrauth (ptr @func_undef, i32 0), ptr @g2
  store ptr ptrauth (ptr @var, i32 0), ptr @g3
  store ptr ptrauth (ptr @var_undef, i32 0), ptr @g4
  ret void
}

define void @_start() {
entry:
  ret void
}

attributes #0 = {"target-features"="+pauth"}

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}
