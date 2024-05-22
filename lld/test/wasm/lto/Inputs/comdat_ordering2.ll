target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; Generated from this C++ code and simplified manually:
;
; int foo();
; inline int unused = foo();
;
; int foo() {
;   return 42;
; }

$unused = comdat any

@unused = linkonce_odr global i32 0, comdat, align 4
@_ZGV6unused = linkonce_odr global i32 0, comdat($unused), align 4
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init, ptr @unused }]

define internal void @__cxx_global_var_init() comdat($unused) {
entry:
  %0 = load i8, ptr @_ZGV6unused, align 4
  %1 = and i8 %0, 1
  %guard.uninitialized = icmp eq i8 %1, 0
  br i1 %guard.uninitialized, label %init.check, label %init.end

init.check:                                       ; preds = %entry
  store i8 1, ptr @_ZGV6unused, align 4
  %call = call i32 @foo()
  store i32 %call, ptr @unused, align 4
  br label %init.end

init.end:                                         ; preds = %init.check, %entry
  ret void
}

define i32 @foo() {
entry:
  ret i32 42
}
