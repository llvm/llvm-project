; ModuleID = '/data00/home/xiongzile/workspace/llvm-worktree/llvm-project/1.cpp'
source_filename = "/data00/home/xiongzile/workspace/llvm-worktree/llvm-project/1.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.C = type { %struct.A.base, [4 x i8], %struct.B.base, i32 }
%struct.A.base = type <{ ptr, i32 }>
%struct.B.base = type <{ ptr, i32 }>

@_ZTV1A = global { [3 x i32] } { [3 x i32] [i32 0, i32 ptrtoint (ptr @_ZTI1A to i32), i32 ptrtoint (ptr @_ZN1A1fEv to i32)] }, align 4
@_ZTVN10__cxxabiv117__class_type_infoE = external global ptr, align 8
@_ZTS1A = global [3 x i8] c"1A\00", align 1
@_ZTI1A = constant { i32, ptr } { i32 ptrtoint (ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16) to i32), ptr @_ZTS1A }, align 8
@_ZTV1B = global { [3 x i32] } { [3 x i32] [i32 0, i32 ptrtoint (ptr @_ZTI1B to i32), i32 ptrtoint (ptr @_ZN1B1gEv to i32)] }, align 4
@_ZTS1B = global [3 x i8] c"1B\00", align 1
@_ZTI1B = constant { i32, ptr } { i32 ptrtoint (ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16) to i32), ptr @_ZTS1B }, align 8
@_ZTV1C = global { [4 x i32], [3 x i32] } { [4 x i32] [i32 0, i32 ptrtoint (ptr @_ZTI1C to i32), i32 ptrtoint (ptr @_ZN1C1fEv to i32), i32 ptrtoint (ptr @_ZN1C1gEv to i32)], [3 x i32] [i32 -16, i32 ptrtoint (ptr @_ZTI1C to i32), i32 ptrtoint (ptr @_ZThn16_N1C1gEv to i32)] }, align 4
@_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global ptr, align 8
@_ZTS1C = global [3 x i8] c"1C\00", align 1
@_ZTI1C = constant { i32, ptr, i32, i32, ptr, i64, ptr, i64 } { i32 ptrtoint (ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 16) to i32), ptr @_ZTS1C, i32 0, i32 2, ptr @_ZTI1A, i64 2, ptr @_ZTI1B, i64 4098 }, align 8

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
  %5 = call ptr @llvm.load.relative.i32(ptr %4, i32 0)
  call void %5(ptr noundef nonnull align 8 dereferenceable(12) %3)
  ret void
}

; Function Attrs: noinline
define dso_local void @_Z14call_through_BP1B(ptr noundef %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = call ptr @llvm.load.relative.i32(ptr %4, i32 0)
  call void %5(ptr noundef nonnull align 8 dereferenceable(12) %3)
  ret void
}

; Function Attrs: noinline
define dso_local void @_Z14call_through_CP1C(ptr noundef %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = call ptr @llvm.load.relative.i32(ptr %4, i32 0)
  call void %5(ptr noundef nonnull align 8 dereferenceable(32) %3)
  %6 = load ptr, ptr %2, align 8
  %7 = load ptr, ptr %6, align 8
  %8 = call ptr @llvm.load.relative.i32(ptr %7, i32 4)
  call void %8(ptr noundef nonnull align 8 dereferenceable(32) %6)
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
  %12 = call ptr @llvm.load.relative.i32(ptr %11, i32 0)
  call void %12(ptr noundef nonnull align 8 dereferenceable(12) %10)
  %13 = load ptr, ptr %4, align 8
  %14 = load ptr, ptr %13, align 8
  %15 = call ptr @llvm.load.relative.i32(ptr %14, i32 0)
  call void %15(ptr noundef nonnull align 8 dereferenceable(12) %13)
  ret void
}

; Function Attrs: noinline
define dso_local void @_Z18construct_and_callv() #0 {
  %1 = alloca %struct.C, i64 1, align 8
  %2 = alloca ptr, i64 1, align 8
  %3 = alloca ptr, i64 1, align 8
  call void @_ZN1CC1Ev(ptr noundef nonnull align 8 dereferenceable(32) %1) #3
  call void @_ZN1C1fEv(ptr noundef nonnull align 8 dereferenceable(32) %1)
  call void @_ZN1C1gEv(ptr noundef nonnull align 8 dereferenceable(32) %1)
  store ptr %1, ptr %2, align 8
  %4 = icmp eq ptr %1, null
  %5 = getelementptr i8, ptr %1, i32 16
  %6 = select i1 %4, ptr %1, ptr %5
  store ptr %6, ptr %3, align 8
  %7 = load ptr, ptr %2, align 8
  %8 = load ptr, ptr %7, align 8
  %9 = call ptr @llvm.load.relative.i32(ptr %8, i32 0)
  call void %9(ptr noundef nonnull align 8 dereferenceable(12) %7)
  %10 = load ptr, ptr %3, align 8
  %11 = load ptr, ptr %10, align 8
  %12 = call ptr @llvm.load.relative.i32(ptr %11, i32 0)
  call void %12(ptr noundef nonnull align 8 dereferenceable(12) %10)
  ret void
}

; Function Attrs: noinline
define linkonce_odr void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1A, i64 8), ptr %3, align 8
  ret void
}

; Function Attrs: noinline
define linkonce_odr void @_ZN1BC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1B, i64 8), ptr %3, align 8
  ret void
}

; Function Attrs: noinline
define linkonce_odr void @_ZN1CC2Ev(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %3) #3
  %4 = getelementptr i8, ptr %3, i32 16
  call void @_ZN1BC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %4) #3
  store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1C, i64 8), ptr %3, align 8
  %5 = getelementptr i8, ptr %3, i32 16
  store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1C, i64 24), ptr %5, align 8
  ret void
}

; Function Attrs: noinline
define linkonce_odr void @_ZN1CC1Ev(ptr noundef nonnull align 8 dereferenceable(32) %0) #0 {
  %2 = alloca ptr, i64 1, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @_ZN1CC2Ev(ptr noundef nonnull align 8 dereferenceable(32) %3) #3
  ret void
}


; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare ptr @llvm.load.relative.i32(ptr, i32) #2

attributes #0 = { noinline "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { noinline }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
