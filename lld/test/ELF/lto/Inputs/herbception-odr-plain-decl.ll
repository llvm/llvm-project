target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; The herbception specifier is not part of the mangled name, so this declares
; and calls the same symbol as herbception-odr-throws.ll with a plain ABI.
declare void @f()

define void @_start() {
entry:
  call void @f()
  ret void
}
