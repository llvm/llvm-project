;; Verify that the target of a wrap is kept by LTO's summary-based global dead
;; stripping if the original symbol is referenced by a linker script or --defsym

; REQUIRES: x86
; RUN: rm -rf %t && split-file %s %t

;; We need a module summary to trigger summary-based global stripping
; RUN: opt -module-summary -o %t/foo.bc %t/foo.ll
; RUN: echo 'alias = __real_foo;' > %t/alias.script
; RUN: ld.lld -shared -o %t/libalias_foo.so %t/foo.bc %t/alias.script --wrap foo
; RUN: llvm-readelf --syms %t/libalias_foo.so | FileCheck --check-prefix=FOO %s

; FOO:     Symbol table '.symtab' contains
; FOO-DAG: [[#]]: [[#%.16x,FOO_VAL:]] 1 FUNC    LOCAL  HIDDEN      [[#]] foo
; FOO-DAG: [[#]]: [[#FOO_VAL]]        0 FUNC    GLOBAL DEFAULT     [[#]] alias

; RUN: opt -module-summary -o %t/wrap_foo.bc %t/wrap_foo.ll
; RUN: ld.lld -shared -o %t/libalias_wrap_foo.so %t/wrap_foo.bc --wrap foo --defsym=alias=foo
; RUN: llvm-readelf --syms %t/libalias_wrap_foo.so | FileCheck --check-prefix=WRAP-FOO %s

; WRAP-FOO:     Symbol table '.symtab' contains
; WRAP-FOO-DAG: [[#]]: [[#%.16x,WRAP_FOO_VAL:]] 1 FUNC    LOCAL  HIDDEN      [[#]] __wrap_foo
; WRAP-FOO-DAG: [[#]]: [[#WRAP_FOO_VAL]]        0 FUNC    GLOBAL DEFAULT     [[#]] alias

;--- foo.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
define hidden void @foo() {
  ret void
}

;; We need a live root to trigger summary-based global stripping
define dso_local void @bar() {
  ret void
}

;--- wrap_foo.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
define hidden void @__wrap_foo() {
  ret void
}

define dso_local void @bar() {
  ret void
}
