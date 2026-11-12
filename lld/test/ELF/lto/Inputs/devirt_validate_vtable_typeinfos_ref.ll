;; Source code:
;; cat > a.h <<'eof'
;; struct A { virtual int foo(); };
;; int bar(A *a);
;; eof
;; cat > b.cc <<'eof'
;; #include "a.h"
;; struct B : A { int foo() { return 2; } };
;; int baz() { B b; return bar(&b); }
;; eof
;; clang++ -flto=thin b.cc -c

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.B = type { %struct.A }
%struct.A = type { ptr }

@_ZTV1B = linkonce_odr dso_local unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1B, ptr @_ZN1B3fooEv] }, !type !0, !type !1, !type !2, !type !3
@_ZTS1B = linkonce_odr dso_local constant [3 x i8] c"1B\00"
@_ZTI1A = external constant ptr
@_ZTI1B = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr null, ptr @_ZTS1B, ptr @_ZTI1A }
@_ZTV1A = external unnamed_addr constant { [3 x ptr] }

define dso_local noundef i32 @_Z3bazv() #0 {
entry:
  %b = alloca %struct.B
  call void @_ZN1BC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %b)
  %call = call noundef i32 @_Z3barP1A(ptr noundef %b)
  ret i32 %call
}

define linkonce_odr dso_local void @_ZN1BC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) #0 {
entry:
  %this.addr = alloca ptr
  store ptr %this, ptr %this.addr
  %this1 = load ptr, ptr %this.addr
  call void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this1)
  store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 2), ptr %this1
  ret void
}

declare i32 @_Z3barP1A(ptr noundef)

define linkonce_odr dso_local void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) #0 {
entry:
  %this.addr = alloca ptr
  store ptr %this, ptr %this.addr
  %this1 = load ptr, ptr %this.addr
  store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 2), ptr %this1
  ret void
}

define linkonce_odr i32 @_ZN1B3fooEv(ptr noundef nonnull align 8 dereferenceable(8) %this) #0 {
entry:
  %this.addr = alloca ptr
  store ptr %this, ptr %this.addr
  %this1 = load ptr, ptr %this.addr
  ret i32 2
}

;; Make sure we don't inline or otherwise optimize out the direct calls.
attributes #0 = { noinline optnone }

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AFivE.virtual"}
!2 = !{i64 16, !"_ZTS1B"}
!3 = !{i64 16, !"_ZTSM1BFivE.virtual"}
