; REQUIRES: x86
; RUN: opt < %s -passes=pseudo-probe -function-sections -o %t.o
; RUN: ld.lld %t.o -shared --lto-pseudo-probe-for-profiling --lto-emit-asm -o - | FileCheck %s
; RUN: ld.lld %t.o -shared -plugin-opt=pseudo-probe-for-profiling --lto-emit-asm -o - | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-scei-ps4"

@g = dso_local global i32 3, align 4

define void @foo(void (i32)* %f) !dbg !4 {
entry:
; CHECK: .pseudoprobe	[[#GUID:]] 1 0 0
; CHECK: .pseudoprobe	[[#GUID]] 2 1 0
  call void %f(i32 1), !dbg !13
  %0 = load i32, i32* @g, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @g, align 4
  ret void
}

; CHECK:      .section .pseudo_probe_desc,"G",@progbits,.pseudo_probe_desc_foo,comdat
; CHECK-NEXT: .quad [[#GUID]]
; CHECK-NEXT: .quad [[#HASH:]]
; CHECK-NEXT: .byte  3
; CHECK-NEXT: .ascii	"foo"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "test.c", directory: "")
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, unit: !0)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !DILocation(line: 2, column: 20, scope: !4)
