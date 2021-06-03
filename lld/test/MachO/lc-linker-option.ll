; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t
;
; RUN: llvm-as %t/framework.ll -o %t/framework.o
; RUN: %lld -lSystem %t/framework.o -o %t/frame
; RUN: llvm-otool -l %t/frame | FileCheck --check-prefix=FRAME %s \
; RUN:  --implicit-check-not LC_LOAD_DYLIB
; FRAME:          cmd LC_LOAD_DYLIB
; FRAME-NEXT: cmdsize
; FRAME-NEXT:    name /usr/lib/libSystem.dylib
; FRAME:          cmd LC_LOAD_DYLIB
; FRAME-NEXT: cmdsize
; FRAME-NEXT:    name /System/Library/Frameworks/CoreFoundation.framework/CoreFoundation

; RUN: llvm-as %t/l.ll -o %t/l.o
;; The dynamic call to _CFBigNumGetInt128 uses dyld_stub_binder,
;; which needs -lSystem from LC_LINKER_OPTION to get resolved.
; RUN: %lld %t/l.o -o %t/l -framework CoreFoundation
; RUN: llvm-otool -l %t/l | FileCheck --check-prefix=LIB %s \
; RUN:  --implicit-check-not LC_LOAD_DYLIB
; LIB:          cmd LC_LOAD_DYLIB
; LIB-NEXT: cmdsize
; LIB-NEXT:    name /System/Library/Frameworks/CoreFoundation.framework/CoreFoundation
; LIB:          cmd LC_LOAD_DYLIB
; LIB-NEXT: cmdsize
; LIB-NEXT:    name /usr/lib/libSystem.dylib

;; Check that we don't create duplicate LC_LOAD_DYLIBs.
; RUN: %lld -lSystem %t/l.o -o %t/l -framework CoreFoundation
; RUN: llvm-otool -l %t/l | FileCheck --check-prefix=FRAME %s \
; RUN:  --implicit-check-not LC_LOAD_DYLIB
;
; RUN: llvm-as %t/invalid.ll -o %t/invalid.o
; RUN: not %lld %t/invalid.o -o /dev/null 2>&1 | FileCheck --check-prefix=INVALID %s
; INVALID: error: -why_load is not allowed in LC_LINKER_OPTION

;--- framework.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!0 = !{!"-framework", !"CoreFoundation"}
!llvm.linker.options = !{!0}

declare void @_CFBigNumGetInt128(...)

define void @main() {
  call void bitcast (void (...)* @_CFBigNumGetInt128 to void ()*)()
  ret void
}

;--- l.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!0 = !{!"-lSystem"}
!llvm.linker.options = !{!0, !0}

declare void @_CFBigNumGetInt128(...)

define void @main() {
  call void bitcast (void (...)* @_CFBigNumGetInt128 to void ()*)()
  ret void
}

;--- invalid.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!0 = !{!"-why_load"}
!llvm.linker.options = !{!0}

define void @main() {
  ret void
}
