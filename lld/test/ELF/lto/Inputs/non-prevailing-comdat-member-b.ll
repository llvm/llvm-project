target datalayout = "e-m:e-p270:32:32:32-p271:32:32:32-p272:64:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZN1AIiED5Ev = comdat any

@_ZN1AIiED1Ev = weak_odr unnamed_addr alias void (ptr), ptr @_ZN1AIiED2Ev

define weak_odr void @_ZN1AIiED2Ev(ptr %this) unnamed_addr comdat($_ZN1AIiED5Ev) {
  ret void
}

define weak_odr void @_ZN1AIiED0Ev(ptr %this) unnamed_addr comdat($_ZN1AIiED5Ev) {
  ret void
}
