; REQUIRES: aarch64

; RUN: llvm-as %s -o %t.o
; RUN: ld.lld %t.o -shared -o %t
; RUN: llvm-readelf -r -x.got %t | FileCheck %s

; CHECK:      Relocation section '.rela.dyn' at offset 0x2a8 contains 2 entries:
; CHECK-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
; CHECK-NEXT: 00000000000203d8  0000000100000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 foo + 0
; CHECK-NEXT: 00000000000203e0  0000000200000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 var + 0

; CHECK:      Hex dump of section '.got':
; CHECK-NEXT: 0x000203d8 00000000 00000080 00000000 000000a0
;;                                      ^^ 0b10000000 bit 63 address diversity = true, bits 61..60 key = IA
;;                                                        ^^ 0b10100000 bit 63 address diversity = true, bits 61..60 key = DA

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

@var = external global ptr

declare void @foo()

define void @bar() #0 {
entry:
  store ptr ptrauth (ptr @foo, i32 0), ptr @var
  ret void
}

define void @_start() {
entry:
  ret void
}

attributes #0 = {"target-features"="+pauth"}

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}
