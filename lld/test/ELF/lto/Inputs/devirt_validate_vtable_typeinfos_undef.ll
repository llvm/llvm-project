; REQUIRES: x86

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTV1B = external unnamed_addr constant { [4 x ptr] }

define linkonce_odr void @_ZN1BC2Ev(ptr %this) #0 {
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  store ptr getelementptr inbounds inrange(-16, 8) ({ [4 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 2), ptr %this1, align 8
  ret void
}

attributes #0 = { noinline optnone }
