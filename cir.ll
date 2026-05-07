; ModuleID = '/data00/home/xiongzile/workspace/llvm-worktree/llvm-project/1.cpp'
source_filename = "/data00/home/xiongzile/workspace/llvm-worktree/llvm-project/1.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.C = type { %struct.A.base, [4 x i8], %struct.B.base, i32 }
%struct.A.base = type <{ ptr, i32 }>
%struct.B.base = type <{ ptr, i32 }>

@_ZTV1A = global { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN1A1fEv] }, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global ptr, align 8
@_ZTS1A = global [3 x i8] c"1A\00", align 1
@_ZTI1A = constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS1A }, align 8
@_ZTV1B = global { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1B, ptr @_ZN1B1gEv] }, align 8
@_ZTS1B = global [3 x i8] c"1B\00", align 1
@_ZTI1B = constant { ptr, ptr } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS1B }, align 8
@_ZTV1C = global { [4 x ptr], [3 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI1C, ptr @_ZN1C1fEv, ptr @_ZN1C1gEv], [3 x ptr] [ptr inttoptr (i64 -16 to ptr), ptr @_ZTI1C, ptr @_ZThn16_N1C1gEv] }, align 8
@_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global ptr, align 8
@_ZTS1C = global [3 x i8] c"1C\00", align 1
@_ZTI1C = constant { ptr, ptr, i32, i32, ptr, i64, ptr, i64 } { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 16), ptr @_ZTS1C, i32 0, i32 2, ptr @_ZTI1A, i64 2, ptr @_ZTI1B, i64 4098 }, align 8

; Function Attrs: noinline
define dso_local void @_ZN1A1fEv(ptr noundef nonnull align 8 dereferenceable(12) %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: noinline
define dso_local void @_ZN1B1gEv(ptr noundef nonnull align 8 dereferenceable(12) %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: noinline
define dso_local void @_ZN1C1fEv(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: noinline
define dso_local void @_ZN1C1gEv(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: noinline
define dso_local void @_ZThn16_N1C1gEv(ptr noundef %0) #1 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr i8, ptr %3, i64 -16
  call void @_ZN1C1gEv(ptr noundef nonnull align 8 dereferenceable(32) %4)
  ret void
}

; Function Attrs: noinline
define dso_local void @_Z14call_through_AP1A(ptr noundef %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr inbounds ptr, ptr %4, i32 0
  %6 = load ptr, ptr %5, align 8
  call void %6(ptr noundef nonnull align 8 dereferenceable(12) %3)
  ret void
}

; Function Attrs: noinline
define dso_local void @_Z14call_through_BP1B(ptr noundef %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr inbounds ptr, ptr %4, i32 0
  %6 = load ptr, ptr %5, align 8
  call void %6(ptr noundef nonnull align 8 dereferenceable(12) %3)
  ret void
}

; Function Attrs: noinline
define dso_local void @_Z14call_through_CP1C(ptr noundef %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr inbounds ptr, ptr %4, i32 0
  %6 = load ptr, ptr %5, align 8
  call void %6(ptr noundef nonnull align 8 dereferenceable(32) %3)
  %7 = load ptr, ptr %2, align 8
  %8 = load ptr, ptr %7, align 8
  %9 = getelementptr inbounds ptr, ptr %8, i32 1
  %10 = load ptr, ptr %9, align 8
  call void %10(ptr noundef nonnull align 8 dereferenceable(32) %7)
  ret void
}

; Function Attrs: noinline
define dso_local void @_Z21direct_qualified_callP1C(ptr noundef %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @_ZN1C1fEv(ptr noundef nonnull align 8 dereferenceable(32) %3)
  %4 = load ptr, ptr %2, align 8
  call void @_ZN1C1gEv(ptr noundef nonnull align 8 dereferenceable(32) %4)
  ret void
}

; Function Attrs: noinline
define dso_local void @_Z10base_castsP1C(ptr noundef %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  %3 = alloca ptr, i64 1, align 8
  %4 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %5 = load ptr, ptr %2, align 8
  store ptr %5, ptr %3, align 8
  %6 = load ptr, ptr %2, align 8
  %7 = icmp eq ptr %6, null
  %8 = getelementptr i8, ptr %6, i32 16
  %9 = select i1 %7, ptr %6, ptr %8
  store ptr %9, ptr %4, align 8
  %10 = load ptr, ptr %3, align 8
  %11 = load ptr, ptr %10, align 8
  %12 = getelementptr inbounds ptr, ptr %11, i32 0
  %13 = load ptr, ptr %12, align 8
  call void %13(ptr noundef nonnull align 8 dereferenceable(12) %10)
  %14 = load ptr, ptr %4, align 8
  %15 = load ptr, ptr %14, align 8
  %16 = getelementptr inbounds ptr, ptr %15, i32 0
  %17 = load ptr, ptr %16, align 8
  call void %17(ptr noundef nonnull align 8 dereferenceable(12) %14)
  ret void
}

; Function Attrs: noinline
define linkonce_odr void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1A, i64 16), ptr %3, align 8
  ret void
}

; Function Attrs: noinline
define linkonce_odr void @_ZN1BC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1B, i64 16), ptr %3, align 8
  ret void
}

; Function Attrs: noinline
define linkonce_odr void @_ZN1CC2Ev(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %3) #2
  %4 = getelementptr i8, ptr %3, i32 16
  call void @_ZN1BC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %4) #2
  store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1C, i64 16), ptr %3, align 8
  %5 = getelementptr i8, ptr %3, i32 16
  store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1C, i64 48), ptr %5, align 8
  ret void
}

; Function Attrs: noinline
define linkonce_odr void @_ZN1CC1Ev(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @_ZN1CC2Ev(ptr noundef nonnull align 8 dereferenceable(32) %3) #2
  ret void
}

; Function Attrs: noinline
define dso_local void @_Z18construct_and_callv() #0 {
  %1 = alloca %struct.C, i64 1, align 8
  %2 = alloca ptr, i64 1, align 8
  %3 = alloca ptr, i64 1, align 8
  call void @_ZN1CC1Ev(ptr noundef nonnull align 8 dereferenceable(32) %1) #2
  call void @_ZN1C1fEv(ptr noundef nonnull align 8 dereferenceable(32) %1)
  call void @_ZN1C1gEv(ptr noundef nonnull align 8 dereferenceable(32) %1)
  store ptr %1, ptr %2, align 8
  %4 = icmp eq ptr %1, null
  %5 = getelementptr i8, ptr %1, i32 16
  %6 = select i1 %4, ptr %1, ptr %5
  store ptr %6, ptr %3, align 8
  %7 = load ptr, ptr %2, align 8
  %8 = load ptr, ptr %7, align 8
  %9 = getelementptr inbounds ptr, ptr %8, i32 0
  %10 = load ptr, ptr %9, align 8
  call void %10(ptr noundef nonnull align 8 dereferenceable(12) %7)
  %11 = load ptr, ptr %3, align 8
  %12 = load ptr, ptr %11, align 8
  %13 = getelementptr inbounds ptr, ptr %12, i32 0
  %14 = load ptr, ptr %13, align 8
  call void %14(ptr noundef nonnull align 8 dereferenceable(12) %11)
  ret void
}

attributes #0 = { noinline "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { noinline }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
