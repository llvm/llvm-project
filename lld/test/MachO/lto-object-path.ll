; REQUIRES: x86
; UNSUPPORTED: system-windows

; RUN: rm -rf %t; mkdir %t
; RUN: opt -thinlto-bc %s -o %t/test.o

; RUN: %lld %t/test.o -o %t/test
; RUN: llvm-nm -pa %t/test | FileCheck %s --check-prefixes CHECK,NOOBJPATH

; RUN: %lld %t/test.o -o %t/test -object_path_lto %t/lto-temps
; RUN: llvm-nm -pa %t/test | FileCheck %s --check-prefixes CHECK,OBJPATH-DIR -DDIR=%t/lto-temps

;; Ensure object path is still used if the cache is used
; RUN: %lld %t/test.o -o %t/test -object_path_lto %t/lto-temps -prune_interval_lto 20 -cache_path_lto %t/cache --thinlto-cache-policy=cache_size_files=1:cache_size_bytes=10
; RUN: llvm-nm -pa %t/test | FileCheck %s --check-prefixes CHECK,OBJPATH-DIR -DDIR=%t/lto-temps
;; And that dsymutil can read the result
; RUN: dsymutil -f -o - %t/test | llvm-dwarfdump - | FileCheck %s --check-prefix=DSYM

;; Test that the object path hardlinks to the cache when possible
;; NB: 
;; 1) Need to delete current object path contents or hardlink will fail.
;; 2) Note the cache policy is different from the test above (which is crafted
;;    to ensure that the cache is pruned immediately; we want the opposite here).
; RUN: rm -rf %t/lto-temps
; RUN: %lld %t/test.o -o %t/test -object_path_lto %t/lto-temps -prune_interval_lto 20 -cache_path_lto %t/cache --thinlto-cache-policy=cache_size_files=10:cache_size_bytes=10000
;; Get the link count.
; RUN: ls -l %t/lto-temps/1.x86_64.lto.o | tr -s ' ' | cut -f 2 -d ' ' | FileCheck %s --check-prefix=HARDLINK

;; check that the object path can be an existing file
; RUN: touch %t/lto-tmp.o
; RUN: %lld %t/test.o -o %t/test -object_path_lto %t/lto-tmp.o
; RUN: llvm-nm -pa %t/test | FileCheck %s --check-prefixes CHECK,OBJPATH-FILE -DFILE=%t/lto-tmp.o


; CHECK:             0000000000000000                - 00 0000    SO /tmp/test.cpp
; NOOBJPATH-NEXT:    0000000000000000                - 03 0001   OSO /tmp/lto.tmp
;; check that modTime is nonzero when `-object_path_lto` is provided
; OBJPATH-DIR-NEXT:  {{[0-9a-f]*[1-9a-f]+[0-9a-f]*}} - 03 0001   OSO [[DIR]]/1.x86_64.lto.o
; OBJPATH-FILE-NEXT: {{[0-9a-f]*[1-9a-f]+[0-9a-f]*}} - 03 0001   OSO [[FILE]]
; CHECK-NEXT:        {{[0-9a-f]+}}                   - 01 0000   FUN _main
; CHECK-NEXT:        0000000000000001                - 00 0000   FUN
; CHECK-NEXT:        0000000000000000                - 01 0000    SO
; CHECK-NEXT:        {{[0-9a-f]+}}                   T _main
; DSYM: DW_AT_name ("test.cpp")
; HARDLINK: 2

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
