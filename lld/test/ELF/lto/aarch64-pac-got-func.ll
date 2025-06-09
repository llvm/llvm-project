; REQUIRES: aarch64

; RUN: llvm-as %s -o %t.o
; RUN: ld.lld %t.o -shared -o %t
; RUN: llvm-readelf -r -x.got %t | FileCheck %s

; CHECK:      Relocation section '.rela.dyn' at offset 0x3d0 contains 8 entries:
; CHECK-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
; CHECK-NEXT: 00000000000210b8  0000000100000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 func_undef + 0
; CHECK-NEXT: 00000000000210c0  0000000200000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 g1 + 0
; CHECK-NEXT: 00000000000210c8  0000000300000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 g2 + 0
; CHECK-NEXT: 00000000000210d0  0000000400000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 g3 + 0
; CHECK-NEXT: 00000000000210d8  0000000500000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 g4 + 0
; CHECK-NEXT: 00000000000210e0  0000000600000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 var_undef + 0
; CHECK-NEXT: 00000000000210a8  0000000700000412 R_AARCH64_AUTH_GLOB_DAT 0000000000010800 func + 0
; CHECK-NEXT: 00000000000210b0  0000000a00000412 R_AARCH64_AUTH_GLOB_DAT 0000000000031400 var + 0

; CHECK:      Hex dump of section '.got':
; CHECK-NEXT: 0x000210a8 00000000 00000080 00000000 000000a0
;;                                      ^^ func: 0b10000000 bit 63 address diversity = true, bits 61..60 key = IA
;;                                                        ^^ var: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA
; CHECK-NEXT: 0x000210b8 00000000 00000080 00000000 000000a0
;;                                      ^^ func_undef: 0b10000000 bit 63 address diversity = true, bits 61..60 key = IA
;;                                                        ^^ g1: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA
; CHECK-NEXT: 0x000210c8 00000000 000000a0 00000000 000000a0
;;                                      ^^ g2: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA
;;                                                        ^^ g3: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA
; CHECK-NEXT: 0x000210d8 00000000 000000a0 00000000 000000a0
;;                                      ^^ g4: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA
;;                                                        ^^ var_undef: 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

@g1 = external global ptr
@g2 = external global ptr
@g3 = external global ptr
@g4 = external global ptr

; Minor codegen changes may influence function sizes and thus move subsequent
; symbols. To prevent accidental changes to symbol addresses, request an
; alignment that is larger than any expected function's size.
;
; Note that it is handy to have a trivial function like _start at the end of
; the .text section, as most offsets of interest point to the dynamic section,
; and one cannot easily control its alignment. On the other hand, the _start
; function almost certainly contains a single ret instruction and is itself
; aligned, making offsets of the subsequent sections predictable.

define void @func() align 1024 {
entry:
  ret void
}
declare void @func_undef()

@var = global i32 42, align 1024
@var_undef = external global i32

define void @bar() #0 align 1024 {
entry:
  store ptr ptrauth (ptr @func, i32 0), ptr @g1
  store ptr ptrauth (ptr @func_undef, i32 0), ptr @g2
  store ptr ptrauth (ptr @var, i32 0), ptr @g3
  store ptr ptrauth (ptr @var_undef, i32 0), ptr @g4
  ret void
}

define void @_start() align 1024 {
entry:
  ret void
}

attributes #0 = {"target-features"="+pauth"}

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}
