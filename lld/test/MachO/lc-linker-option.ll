; REQUIRES: x86, shell
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

;; We are testing this because we want to check a dangling string reference problem (see https://reviews.llvm.org/D111706).
;; To trigger this problem, we need to create a framework that is an archive,
;; and it needs to contain a symbol starting with OBJC_CLASS_$.
;; The bug is triggered as the linker loads this framework twice via LC_LINKER_OPTION.
;; When the linker adds this framework, it will fail to map the path of framework to this archive due to dangling reference.
;; Therefore, it will load the framework twice, and if there is any symbol with OBJC_CLASS_$ prefix with forceLoadObjC enabled,
;; the linker will fetch this symbol twice, which leads to a duplicate symbol error.
; RUN: llc %t/foo.ll -o %t/foo.o -filetype=obj
; RUN: mkdir -p %t/foo.framework/Versions/A
; RUN: llvm-ar rcs %t/foo.framework/Versions/A/foo %t/foo.o
; RUN: ln -sf A %t/foo.framework/Versions/Current
; RUN: ln -sf Versions/Current/foo %t/foo.framework/foo
; RUN: llc %t/objfile1.ll -o %t/objfile1.o -filetype=obj
; RUN: llc %t/objfile2.ll -o %t/objfile2.o -filetype=obj
; RUN: llc %t/main.ll -o %t/main.o -filetype=obj
; RUN: %lld -demangle -ObjC %t/objfile1.o %t/objfile2.o %t/main.o -o %t/main.out -arch x86_64 -platform_version macos 11.0.0 0.0.0 -F%t

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

!0 = !{!"-framework", !"foo"}
!llvm.linker.options = !{!0}

;--- objfile2.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!0 = !{!"-framework", !"foo"}
!llvm.linker.options = !{!0}

;--- main.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @main() {
  ret void
}

;--- foo.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%0 = type opaque
%struct._class_t = type {}

@"OBJC_CLASS_$_TestClass" = global %struct._class_t {}, section "__DATA, __objc_data", align 8
