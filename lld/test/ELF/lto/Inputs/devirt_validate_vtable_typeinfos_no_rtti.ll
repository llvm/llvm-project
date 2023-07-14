; REQUIRES: x86

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { ptr }
%struct.Native = type { %struct.A }

@_ZTV6Native = linkonce_odr unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZN1A1nEi, ptr @_ZN6Native1fEi] }

define linkonce_odr i32 @_ZN6Native1fEi(ptr %this, i32 %a) #0 {
   ret i32 1;
}

define linkonce_odr i32 @_ZN1A1nEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

attributes #0 = { noinline optnone }
