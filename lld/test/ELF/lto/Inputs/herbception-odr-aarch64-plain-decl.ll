target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

;; The herbception specifier is not part of the mangled name, so this declares
;; and calls the same symbol as herbception-odr-aarch64-throws.ll with a plain
;; ABI.
declare void @f()

define void @_start() {
entry:
  call void @f()
  ret void
}
