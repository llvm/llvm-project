target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { ptr }

@_ZTV1A = unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr undef, ptr @_ZN1A1fEi, ptr @_ZN1A1nEi] }, !type !0, !vcall_visibility !1

define i32 @_ZN1A1fEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

define i32 @_ZN1A1nEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

attributes #0 = { noinline optnone }

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 0}
