; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

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
;; The reference to __cxa_allocate_exception will require -lc++ from
;; LC_LINKER_OPTION to get resolved.
; RUN: %lld %t/l.o -o %t/l -framework CoreFoundation
; RUN: llvm-otool -l %t/l | FileCheck --check-prefix=LIB %s \
; RUN:  --implicit-check-not LC_LOAD_DYLIB
; LIB:          cmd LC_LOAD_DYLIB
; LIB-NEXT: cmdsize
; LIB-NEXT:    name /System/Library/Frameworks/CoreFoundation.framework/CoreFoundation
; LIB:          cmd LC_LOAD_DYLIB
; LIB-NEXT: cmdsize
; LIB-NEXT:    name /usr/lib/libSystem.dylib
; LIB:          cmd LC_LOAD_DYLIB
; LIB-NEXT: cmdsize
; LIB-NEXT:    name /usr/lib/libc++abi.dylib

;; Check that we don't create duplicate LC_LOAD_DYLIBs.
; RUN: %lld -lSystem %t/l.o -o %t/l -framework CoreFoundation
; RUN: llvm-otool -l %t/l | FileCheck --check-prefix=LIB2 %s \
; RUN:  --implicit-check-not LC_LOAD_DYLIB
; LIB2:          cmd LC_LOAD_DYLIB
; LIB2-NEXT: cmdsize
; LIB2-NEXT:    name /usr/lib/libSystem.dylib
; LIB2:          cmd LC_LOAD_DYLIB
; LIB2-NEXT: cmdsize
; LIB2-NEXT:    name /System/Library/Frameworks/CoreFoundation.framework/CoreFoundation
; LIB2:          cmd LC_LOAD_DYLIB
; LIB2-NEXT: cmdsize
; LIB2-NEXT:    name /usr/lib/libc++abi.dylib

; RUN: llvm-as %t/invalid.ll -o %t/invalid.o
; RUN: not %lld %t/invalid.o -o /dev/null 2>&1 | FileCheck --check-prefix=INVALID %s
; INVALID: error: -why_load is not allowed in LC_LINKER_OPTION

;; This is a regression test for a dangling string reference issue that occurred
;; when loading an archive-based framework via LC_LINKER_OPTION (see
;; D111706). Prior to the fix, this would trigger a heap-use-after-free when run
;; under ASAN.
; RUN: llc %t/foo.ll -o %t/foo.o -filetype=obj
; RUN: mkdir -p %t/Foo.framework
;; In a proper framework, this is technically supposed to be a symlink to the
;; actual archive at Foo.framework/Versions/Current, but we skip that here so
;; that this test can run on Windows.
; RUN: llvm-ar rcs %t/Foo.framework/Foo %t/foo.o
; RUN: llc %t/objfile1.ll -o %t/objfile1.o -filetype=obj
; RUN: llc %t/main.ll -o %t/main.o -filetype=obj
; RUN: %lld %t/objfile1.o %t/main.o -o /dev/null -F%t

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
!1 = !{!"-lc++"}
!llvm.linker.options = !{!0, !0, !1}

declare void @_CFBigNumGetInt128(...)
declare i8* @__cxa_allocate_exception(i64)

define void @main() {
  call void bitcast (void (...)* @_CFBigNumGetInt128 to void ()*)()
  call i8* @__cxa_allocate_exception(i64 4)
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

;--- objfile1.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!0 = !{!"-framework", !"Foo"}
!llvm.linker.options = !{!0}

;--- main.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @main() {
  ret void
}

!0 = !{!"-framework", !"Foo"}
!llvm.linker.options = !{!0}

;--- foo.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  ret void
}
