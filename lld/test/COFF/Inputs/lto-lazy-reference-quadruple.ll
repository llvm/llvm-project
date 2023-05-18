target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

define double @quadruple(double %x) {
entry:
  ; The symbol __real@40800000 is used to materialize the 4.0 constant.
  %mul = fmul double %x, 4.0
  ret double %mul
}


declare void @dummy()
define void @f() {
  call void @dummy()
  ret void
}
