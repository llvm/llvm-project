target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

%class.baz = type { %class.bar }
%class.bar = type { ptr }

$"\01?x@bar@@UEBA_NXZ" = comdat any

$"\01??_7baz@@6B@" = comdat any

$"\01??_Gbaz@@UEAAPEAXI@Z" = comdat any

@"\01??_7baz@@6B@" = linkonce_odr unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"\01??_Gbaz@@UEAAPEAXI@Z", ptr @"\01?x@bar@@UEBA_NXZ"] }, comdat, !type !0, !type !1

define void @"\01?qux@@YAXXZ"() local_unnamed_addr {
  ret void
}

define linkonce_odr ptr @"\01??_Gbaz@@UEAAPEAXI@Z"(ptr %this, i32 %should_call_delete) unnamed_addr comdat {
  ret ptr null
}

define linkonce_odr zeroext i1 @"\01?x@bar@@UEBA_NXZ"(ptr %this) unnamed_addr comdat {
  ret i1 false
}

!0 = !{i64 0, !"?AVbar@@"}
!1 = !{i64 0, !"?AVbaz@@"}
