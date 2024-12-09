target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

$f1 = comdat any
$f2 = comdat any

define weak_odr void @bah() {
  ret void
}

define linkonce_odr void @f1() local_unnamed_addr comdat {
  ret void
}
define weak_odr void @f2() local_unnamed_addr comdat {
  ret void
}