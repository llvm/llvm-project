; REQUIRES: x86

; RUN: rm -rf %t; mkdir %t
; RUN: opt -module-hash -module-summary %s -o %t/foo.o
; RUN: %lld -cache_path_lto %t/cache -o %t/test %t/foo.o

;; Check that dsymutil is able to read the .o file out of the thinlto cache.
; RUN: dsymutil -f -o - %t/test | llvm-dwarfdump - | FileCheck %s
; CHECK: DW_AT_name ("test.cpp")


target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @main() #0 !dbg !4 {
  ret void
}

!llvm.module.flags = !{ !0, !1 }
!llvm.dbg.cu = !{!2}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, emissionKind: FullDebug)
!3 = !DIFile(filename: "test.cpp", directory: "/tmp")
!4 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 1, type: !5, scopeLine: 1, unit: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{}
