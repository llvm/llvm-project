; REQUIRES: aarch64

; RUN: llvm-as %s -o %t.o
; RUN: ld.lld %t.o -shared -o %t
; RUN: llvm-readelf -r -x.got %t | FileCheck %s

; CHECK:      Relocation section '.rela.dyn' at offset 0x3d0 contains 8 entries:
; CHECK-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
; CHECK-NEXT: 00000000000206a0  0000000100000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 func_undef + 0
; CHECK-NEXT: 00000000000206a8  0000000200000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 g1 + 0
; CHECK-NEXT: 00000000000206b0  0000000300000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 g2 + 0
; CHECK-NEXT: 00000000000206b8  0000000400000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 g3 + 0
; CHECK-NEXT: 00000000000206c0  0000000500000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 g4 + 0
; CHECK-NEXT: 00000000000206c8  0000000600000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 var_undef + 0
; CHECK-NEXT: 0000000000020690  0000000700000412 R_AARCH64_AUTH_GLOB_DAT 0000000000010490 func + 0
; CHECK-NEXT: 0000000000020698  0000000a00000412 R_AARCH64_AUTH_GLOB_DAT 00000000000306d0 var + 0

; CHECK:      Hex dump of section '.got':
; CHECK-NEXT: 0x00020690 00000000 00000080 00000000 000000a0
;;                                      ^^ func: 0b10000000 bit 63 address diversity = true, bits 61..60 key = IA
;;                                                        ^^ var: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA
; CHECK-NEXT: 0x000206a0 00000000 00000080 00000000 000000a0
;;                                      ^^ func_undef: 0b10000000 bit 63 address diversity = true, bits 61..60 key = IA
;;                                                        ^^ g1: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA
; CHECK-NEXT: 0x000206b0 00000000 000000a0 00000000 000000a0
;;                                      ^^ g2: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA
;;                                                        ^^ g3: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA
; CHECK-NEXT: 0x000206c0 00000000 000000a0 00000000 000000a0
;;                                      ^^ g4: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA
;;                                                        ^^ var_undef: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA

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
