target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

$ext_and_ext = comdat any
$lo_and_ext = comdat any
$lo_and_wo = comdat any
$wo_and_lo = comdat any

declare void @foo(i64)

define weak_odr void @bah() {
  ret void
}

define void @ext_and_ext() local_unnamed_addr comdat {
  call void @foo(i64 2)
  ret void
}

define linkonce_odr void @lo_and_ext() local_unnamed_addr comdat {
  call void @foo(i64 2)
  ret void
}

define weak_odr void @lo_and_wo() local_unnamed_addr comdat {
  ret void
}

define linkonce_odr void @wo_and_lo() local_unnamed_addr comdat {
  ret void
}
