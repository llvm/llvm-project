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

; RUN: not %lld %t/framework.o -o %t/frame_no_autolink -ignore_auto_link 2>&1 | FileCheck --check-prefix=NO_AUTOLINK %s
; RUN: not %lld %t/framework.o -o %t/frame_no_autolink --ignore-auto-link-option CoreFoundation 2>&1 | FileCheck --check-prefix=NO_AUTOLINK %s
; RUN: not %lld %t/framework.o -o %t/frame_no_autolink --ignore-auto-link-option=CoreFoundation 2>&1 | FileCheck --check-prefix=NO_AUTOLINK %s
; NO_AUTOLINK: error: undefined symbol: __CFBigNumGetInt128

; RUN: llvm-as %t/l.ll -o %t/l.o
;; The dynamic call to _CFBigNumGetInt128 uses dyld_stub_binder,
;; which needs -lSystem from LC_LINKER_OPTION to get resolved.
;; The reference to __cxa_allocate_exception will require -lc++ from
;; LC_LINKER_OPTION to get resolved.
; RUN: %no-lsystem-lld %t/l.o -o %t/l -framework CoreFoundation
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
; RUN: %no-lsystem-lld -lSystem %t/l.o -o %t/l -framework CoreFoundation
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
; RUN: llc %t/load-framework-foo.ll -o %t/load-framework-foo.o -filetype=obj
; RUN: llc %t/main.ll -o %t/main.o -filetype=obj
; RUN: %lld %t/load-framework-foo.o %t/main.o -o %t/main -F%t
; RUN: llvm-objdump --macho --syms %t/main | FileCheck %s --check-prefix=SYMS

;; Make sure -all_load and -ObjC have no effect on libraries loaded via
;; LC_LINKER_OPTION flags.
; RUN: llc %t/load-library-foo.ll -o %t/load-library-foo.o -filetype=obj
; RUN: llvm-ar rcs %t/libfoo.a %t/foo.o
; RUN: %lld -all_load -ObjC %t/load-framework-foo.o %t/load-library-foo.o \
; RUN:   %t/main.o -o %t/main -F%t -L%t
; RUN: llvm-objdump --macho --syms %t/main | FileCheck %s --check-prefix=SYMS

;; Note that _OBJC_CLASS_$_TestClass is *not* included here.
; SYMS:       SYMBOL TABLE:
; SYMS-NEXT:  g     F __TEXT,__text _main
; SYMS-NEXT:  g     F __TEXT,__text __mh_execute_header
; SYMS-NEXT:  *UND* dyld_stub_binder
; SYMS-EMPTY:

;; Make sure -all_load has effect when libraries are loaded via LC_LINKER_OPTION flags and explicitly passed as well
; RUN: %lld -all_load %t/load-framework-foo.o %t/load-library-foo.o %t/main.o -o %t/main -F%t -L%t -lfoo
; RUN: llvm-objdump --macho --syms %t/main | FileCheck %s --check-prefix=SYMS_ALL_LOAD

;; Note that _OBJC_CLASS_$_TestClass is *included* here.
; SYMS_ALL_LOAD:       SYMBOL TABLE:
; SYMS_ALL_LOAD-NEXT:  g     F __TEXT,__text _main
; SYMS_ALL_LOAD-NEXT:  g     O __DATA,__objc_data _OBJC_CLASS_$_TestClass
; SYMS_ALL_LOAD-NEXT:  g     F __TEXT,__text __mh_execute_header
; SYMS_ALL_LOAD-NEXT:  *UND* dyld_stub_binder
; SYMS_ALL_LOAD-EMPTY:

;; Make sure -force_load has effect when libraries are loaded via LC_LINKER_OPTION flags and explicitly passed as well
; RUN: %lld %t/load-library-foo.o %t/main.o -o %t/main -F%t -L%t -force_load %t/libfoo.a
; RUN: llvm-objdump --macho --syms %t/main | FileCheck %s --check-prefix=SYMS_FORCE_LOAD

;; Note that _OBJC_CLASS_$_TestClass is *included* here.
; SYMS_FORCE_LOAD:       SYMBOL TABLE:
; SYMS_FORCE_LOAD-NEXT:  g     F __TEXT,__text _main
; SYMS_FORCE_LOAD-NEXT:  g     O __DATA,__objc_data _OBJC_CLASS_$_TestClass
; SYMS_FORCE_LOAD-NEXT:  g     F __TEXT,__text __mh_execute_header
; SYMS_FORCE_LOAD-NEXT:  *UND* dyld_stub_binder
; SYMS_FORCE_LOAD-EMPTY:

;; Make sure -ObjC has effect when frameworks are loaded via LC_LINKER_OPTION flags and explicitly passed as well
; RUN: %lld -ObjC %t/load-framework-foo.o %t/load-library-foo.o %t/main.o -o %t/main -F%t -L%t -framework Foo
; RUN: llvm-objdump --macho --syms %t/main | FileCheck %s --check-prefix=SYMS_OBJC_LOAD

;; Note that _OBJC_CLASS_$_TestClass is *included* here.
; SYMS_OBJC_LOAD:       SYMBOL TABLE:
; SYMS_OBJC_LOAD-NEXT:  g     F __TEXT,__text _main
; SYMS_OBJC_LOAD-NEXT:  g     O __DATA,__objc_data _OBJC_CLASS_$_TestClass
; SYMS_OBJC_LOAD-NEXT:  g     F __TEXT,__text __mh_execute_header
; SYMS_OBJC_LOAD-NEXT:  *UND* dyld_stub_binder
; SYMS_OBJC_LOAD-EMPTY:

;; Make sure that frameworks containing object files or bitcode instead of
;; dylibs or archives do not cause duplicate symbol errors
; RUN: mkdir -p %t/Foo.framework
; RUN: llc --filetype=obj %t/foo.ll -o %t/Foo.framework/Foo
; RUN: llc --filetype=obj %t/load-framework-twice.ll -o %t/main
;; Order of the object with the LC_LINKER_OPTION vs -framework arg is important.
; RUN: %lld %t/main -F %t -framework Foo -framework Foo -o /dev/null
; RUN: %lld -F %t -framework Foo -framework Foo %t/main -o /dev/null

; RUN: llvm-as %t/foo.ll -o %t/Foo.framework/Foo
; RUN: llvm-as %t/load-framework-twice.ll -o %t/main
;; Order of the object with the LC_LINKER_OPTION vs -framework arg is important.
; RUN: %lld %t/main -F %t -framework Foo -framework Foo -o /dev/null
; RUN: %lld -F %t -framework Foo -framework Foo %t/main -o /dev/null

;--- framework.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!0 = !{!"-framework", !"CoreFoundation"}
!llvm.linker.options = !{!0}

declare void @_CFBigNumGetInt128(...)

define void @main() {
  call void @_CFBigNumGetInt128()
  ret void
}

;--- l.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!0 = !{!"-lSystem"}
!1 = !{!"-lc++"}
!llvm.linker.options = !{!0, !0, !1}

declare void @_CFBigNumGetInt128(...)
declare ptr @__cxa_allocate_exception(i64)

define void @main() {
  call void @_CFBigNumGetInt128()
  call ptr @__cxa_allocate_exception(i64 4)
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

;--- load-framework-foo.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!0 = !{!"-framework", !"Foo"}
!llvm.linker.options = !{!0}

;--- load-framework-twice.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!0 = !{!"-framework", !"Foo"}
!llvm.linker.options = !{!0, !0}

define void @main() {
  ret void
}

;--- load-library-foo.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!0 = !{!"-lfoo"}
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

%struct._class_t = type {}
@"OBJC_CLASS_$_TestClass" = global %struct._class_t {}, section "__DATA, __objc_data", align 8
