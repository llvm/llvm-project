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

; RUN: not %lld %t/framework.o -o %t/frame_no_autolink -ignore_auto_link 2>&1 | FileCheck --check-prefix=NO-AUTOLINK %s
; RUN: not %lld %t/framework.o -o %t/frame_no_autolink --ignore-auto-link-option CoreFoundation 2>&1 | FileCheck --check-prefix=NO-AUTOLINK %s
; RUN: not %lld %t/framework.o -o %t/frame_no_autolink --ignore-auto-link-option=CoreFoundation 2>&1 | FileCheck --check-prefix=NO-AUTOLINK %s
; NO-AUTOLINK: error: undefined symbol: __CFBigNumGetInt128

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
; RUN: llc %t/load-framework-undefined-symbol.ll -o %t/load-framework-undefined-symbol.o -filetype=obj
; RUN: llc %t/load-missing.ll -o %t/load-missing.o -filetype=obj
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
; RUN: llvm-objdump --macho --syms %t/main | FileCheck %s --check-prefix=SYMS-ALL-LOAD

;; Note that _OBJC_CLASS_$_TestClass is *included* here.
; SYMS-ALL-LOAD:       SYMBOL TABLE:
; SYMS-ALL-LOAD-NEXT:  g     F __TEXT,__text _main
; SYMS-ALL-LOAD-NEXT:  g     O __DATA,__objc_data _OBJC_CLASS_$_TestClass
; SYMS-ALL-LOAD-NEXT:  g     F __TEXT,__text __mh_execute_header
; SYMS-ALL-LOAD-NEXT:  *UND* dyld_stub_binder
; SYMS-ALL-LOAD-EMPTY:

;; Make sure -force_load has effect when libraries are loaded via LC_LINKER_OPTION flags and explicitly passed as well
; RUN: %lld %t/load-library-foo.o %t/main.o -o %t/main -F%t -L%t -force_load %t/libfoo.a
; RUN: llvm-objdump --macho --syms %t/main | FileCheck %s --check-prefix=SYMS-FORCE-LOAD

;; Note that _OBJC_CLASS_$_TestClass is *included* here.
; SYMS-FORCE-LOAD:       SYMBOL TABLE:
; SYMS-FORCE-LOAD-NEXT:  g     F __TEXT,__text _main
; SYMS-FORCE-LOAD-NEXT:  g     O __DATA,__objc_data _OBJC_CLASS_$_TestClass
; SYMS-FORCE-LOAD-NEXT:  g     F __TEXT,__text __mh_execute_header
; SYMS-FORCE-LOAD-NEXT:  *UND* dyld_stub_binder
; SYMS-FORCE-LOAD-EMPTY:

;; Make sure -ObjC has effect when frameworks are loaded via LC_LINKER_OPTION flags and explicitly passed as well
; RUN: %lld -ObjC %t/load-framework-foo.o %t/load-library-foo.o %t/main.o -o %t/main -F%t -L%t -framework Foo
; RUN: llvm-objdump --macho --syms %t/main | FileCheck %s --check-prefix=SYMS-OBJC-LOAD

;; Note that _OBJC_CLASS_$_TestClass is *included* here.
; SYMS-OBJC-LOAD:       SYMBOL TABLE:
; SYMS-OBJC-LOAD-NEXT:  g     F __TEXT,__text _main
; SYMS-OBJC-LOAD-NEXT:  g     O __DATA,__objc_data _OBJC_CLASS_$_TestClass
; SYMS-OBJC-LOAD-NEXT:  g     F __TEXT,__text __mh_execute_header
; SYMS-OBJC-LOAD-NEXT:  *UND* dyld_stub_binder
; SYMS-OBJC-LOAD-EMPTY:

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

;; Checks that "framework not found" errors from LC_LINKER_OPTIONS are not
;; emitted unless the link fails or --strict-auto-link is passed.
; RUN: %lld -ObjC %t/load-framework-foo.o %t/main.o -o %t/main-no-foo.out
; RUN: llvm-objdump --macho --syms %t/main-no-foo.out | FileCheck %s --check-prefix=SYMS-NO-FOO
; RUN: not %lld --strict-auto-link -ObjC %t/load-missing.o %t/main.o -o %t/main-no-foo.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=MISSING-AUTO-LINK
; RUN: %no-fatal-warnings-lld --strict-auto-link -ObjC %t/load-missing.o %t/main.o -o %t/main-no-foo.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=MISSING-AUTO-LINK
; RUN: not %lld -ObjC %t/load-framework-undefined-symbol.o %t/load-missing.o %t/main.o -o %t/main-no-foo.out 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=UNDEFINED-SYMBOL,MISSING-AUTO-LINK

;; Verify that nothing from the framework is included.
; SYMS-NO-FOO:       SYMBOL TABLE:
; SYMS-NO-FOO-NEXT:  g     F __TEXT,__text _main
; SYMS-NO-FOO-NOT:   g     O __DATA,__objc_data _OBJC_CLASS_$_TestClass

; UNDEFINED-SYMBOL: undefined symbol: __SomeUndefinedSymbol
; MISSING-AUTO-LINK: {{.+}}load-missing.o: auto-linked framework not found for -framework Foo
; MISSING-AUTO-LINK: {{.+}}load-missing.o: auto-linked library not found for -lBar

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

;--- load-missing.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

!0 = !{!"-framework", !"Foo"}
!1 = !{!"-lBar"}
!llvm.linker.options = !{!0, !1}

;--- load-framework-undefined-symbol.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare void @_SomeUndefinedSymbol(...)
define void @foo() {
  call void @_SomeUndefinedSymbol()
  ret void
}

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
