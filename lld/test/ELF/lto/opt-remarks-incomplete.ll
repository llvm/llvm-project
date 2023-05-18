; REQUIRES: x86
;; Ensure passing --plugin-opt=emit-llvm to lld, LTO should not emit
;; incomplete optimization remarks for dead functions.

; RUN: split-file %s %t.dir
; RUN: opt -module-summary %t.dir/main.ll -o %t1.o
; RUN: opt -module-summary %t.dir/other.ll -o %t2.o

; RUN: rm -f %t.yaml
; RUN: ld.lld --plugin-opt=emit-llvm --opt-remarks-filename %t.yaml %t1.o %t2.o -o %t
; RUN: FileCheck %s --check-prefix=REMARK < %t.yaml

; REMARK:      Pass:            lto
; REMARK-NEXT: Name:            deadfunction
; REMARK-NEXT: DebugLoc:        { File: test.c, Line: 4, Column: 0 }
; REMARK-NEXT: Function:        dead2
; REMARK-NEXT: Args:
; REMARK-NEXT:   - Function:        dead2
; REMARK-NEXT:     DebugLoc:        { File: test.c, Line: 4, Column: 0 }
; REMARK-NEXT:   - String:          ' not added to the combined module '

#--- main.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @_start() {
  call void @live1()
  ret void
}

declare void @live1()

define void @live2() {
  ret void
}

define void @dead2() !dbg !7 {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"ThinLTO", i32 0}
!7 = distinct !DISubprogram(name: "dead2", scope: !1, file: !1, line: 4, type: !8, isLocal: false, isDefinition: true, scopeLine: 4, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)

#--- other.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @live1() {
  call void @live2()
  ret void
}

declare void @live2()

define void @dead1() {
  call void @dead2()
  ret void
}

declare void @dead2()
