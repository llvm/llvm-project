; ModuleID = '1.cpp'
source_filename = "1.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.C = type { %struct.A.base, [4 x i8], %struct.B.base, i32 }
%struct.A.base = type <{ ptr, i32 }>
%struct.B.base = type <{ ptr, i32 }>

$_ZN1CC1Ev = comdat any

$_ZN1CC2Ev = comdat any

$_ZN1AC2Ev = comdat any

$_ZN1BC2Ev = comdat any

@_ZTV1A = unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN1A1fEv] }, align 8
@_ZTI1A = constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A }, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTS1A = constant [3 x i8] c"1A\00", align 1
@_ZTV1B = unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1B, ptr @_ZN1B1gEv] }, align 8
@_ZTI1B = constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1B }, align 8
@_ZTS1B = constant [3 x i8] c"1B\00", align 1
@_ZTV1C = unnamed_addr constant { [4 x ptr], [3 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI1C, ptr @_ZN1C1fEv, ptr @_ZN1C1gEv], [3 x ptr] [ptr inttoptr (i64 -16 to ptr), ptr @_ZTI1C, ptr @_ZThn16_N1C1gEv] }, align 8
@_ZTI1C = constant { ptr, ptr, i32, i32, ptr, i64, ptr, i64 } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2), ptr @_ZTS1C, i32 0, i32 2, ptr @_ZTI1A, i64 2, ptr @_ZTI1B, i64 4098 }, align 8
@_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global [0 x ptr]
@_ZTS1C = constant [3 x i8] c"1C\00", align 1

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_ZN1A1fEv(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_ZN1B1gEv(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_ZN1C1fEv(ptr noundef nonnull align 8 dereferenceable(32) %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_ZN1C1gEv(ptr noundef nonnull align 8 dereferenceable(32) %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone
define dso_local void @_ZThn16_N1C1gEv(ptr noundef %this) unnamed_addr #1 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %0 = getelementptr inbounds i8, ptr %this1, i64 -16
  tail call void @_ZN1C1gEv(ptr noundef nonnull align 8 dereferenceable(32) %0)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z14call_through_AP1A(ptr noundef %pa) #0 {
entry:
  %pa.addr = alloca ptr, align 8
  store ptr %pa, ptr %pa.addr, align 8
  %0 = load ptr, ptr %pa.addr, align 8
  %vtable = load ptr, ptr %0, align 8
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 0
  %1 = load ptr, ptr %vfn, align 8
  call void %1(ptr noundef nonnull align 8 dereferenceable(12) %0)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z14call_through_BP1B(ptr noundef %pb) #0 {
entry:
  %pb.addr = alloca ptr, align 8
  store ptr %pb, ptr %pb.addr, align 8
  %0 = load ptr, ptr %pb.addr, align 8
  %vtable = load ptr, ptr %0, align 8
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 0
  %1 = load ptr, ptr %vfn, align 8
  call void %1(ptr noundef nonnull align 8 dereferenceable(12) %0)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z14call_through_CP1C(ptr noundef %pc) #0 {
entry:
  %pc.addr = alloca ptr, align 8
  store ptr %pc, ptr %pc.addr, align 8
  %0 = load ptr, ptr %pc.addr, align 8
  %vtable = load ptr, ptr %0, align 8
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 0
  %1 = load ptr, ptr %vfn, align 8
  call void %1(ptr noundef nonnull align 8 dereferenceable(32) %0)
  %2 = load ptr, ptr %pc.addr, align 8
  %vtable1 = load ptr, ptr %2, align 8
  %vfn2 = getelementptr inbounds ptr, ptr %vtable1, i64 1
  %3 = load ptr, ptr %vfn2, align 8
  call void %3(ptr noundef nonnull align 8 dereferenceable(32) %2)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z21direct_qualified_callP1C(ptr noundef %pc) #0 {
entry:
  %pc.addr = alloca ptr, align 8
  store ptr %pc, ptr %pc.addr, align 8
  %0 = load ptr, ptr %pc.addr, align 8
  call void @_ZN1C1fEv(ptr noundef nonnull align 8 dereferenceable(32) %0)
  %1 = load ptr, ptr %pc.addr, align 8
  call void @_ZN1C1gEv(ptr noundef nonnull align 8 dereferenceable(32) %1)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z10base_castsP1C(ptr noundef %pc) #0 {
entry:
  %pc.addr = alloca ptr, align 8
  %pa = alloca ptr, align 8
  %pb = alloca ptr, align 8
  store ptr %pc, ptr %pc.addr, align 8
  %0 = load ptr, ptr %pc.addr, align 8
  store ptr %0, ptr %pa, align 8
  %1 = load ptr, ptr %pc.addr, align 8
  %2 = icmp eq ptr %1, null
  br i1 %2, label %cast.end, label %cast.notnull

cast.notnull:                                     ; preds = %entry
  %add.ptr = getelementptr inbounds i8, ptr %1, i64 16
  br label %cast.end

cast.end:                                         ; preds = %cast.notnull, %entry
  %cast.result = phi ptr [ %add.ptr, %cast.notnull ], [ null, %entry ]
  store ptr %cast.result, ptr %pb, align 8
  %3 = load ptr, ptr %pa, align 8
  %vtable = load ptr, ptr %3, align 8
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 0
  %4 = load ptr, ptr %vfn, align 8
  call void %4(ptr noundef nonnull align 8 dereferenceable(12) %3)
  %5 = load ptr, ptr %pb, align 8
  %vtable1 = load ptr, ptr %5, align 8
  %vfn2 = getelementptr inbounds ptr, ptr %vtable1, i64 0
  %6 = load ptr, ptr %vfn2, align 8
  call void %6(ptr noundef nonnull align 8 dereferenceable(12) %5)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z18construct_and_callv() #0 {
entry:
  %obj = alloca %struct.C, align 8
  %pa = alloca ptr, align 8
  %pb = alloca ptr, align 8
  call void @_ZN1CC1Ev(ptr noundef nonnull align 8 dereferenceable(32) %obj) #2
  call void @_ZN1C1fEv(ptr noundef nonnull align 8 dereferenceable(32) %obj)
  call void @_ZN1C1gEv(ptr noundef nonnull align 8 dereferenceable(32) %obj)
  store ptr %obj, ptr %pa, align 8
  %0 = icmp eq ptr %obj, null
  br i1 %0, label %cast.end, label %cast.notnull

cast.notnull:                                     ; preds = %entry
  %add.ptr = getelementptr inbounds i8, ptr %obj, i64 16
  br label %cast.end

cast.end:                                         ; preds = %cast.notnull, %entry
  %cast.result = phi ptr [ %add.ptr, %cast.notnull ], [ null, %entry ]
  store ptr %cast.result, ptr %pb, align 8
  %1 = load ptr, ptr %pa, align 8
  %vtable = load ptr, ptr %1, align 8
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 0
  %2 = load ptr, ptr %vfn, align 8
  call void %2(ptr noundef nonnull align 8 dereferenceable(12) %1)
  %3 = load ptr, ptr %pb, align 8
  %vtable1 = load ptr, ptr %3, align 8
  %vfn2 = getelementptr inbounds ptr, ptr %vtable1, i64 0
  %4 = load ptr, ptr %vfn2, align 8
  call void %4(ptr noundef nonnull align 8 dereferenceable(12) %3)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr void @_ZN1CC1Ev(ptr noundef nonnull align 8 dereferenceable(32) %this) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  call void @_ZN1CC2Ev(ptr noundef nonnull align 8 dereferenceable(32) %this1) #2
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr void @_ZN1CC2Ev(ptr noundef nonnull align 8 dereferenceable(32) %this) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  call void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this1) #2
  %0 = getelementptr inbounds i8, ptr %this1, i64 16
  call void @_ZN1BC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %0) #2
  store ptr getelementptr inbounds inrange(-16, 16) ({ [4 x ptr], [3 x ptr] }, ptr @_ZTV1C, i32 0, i32 0, i32 2), ptr %this1, align 8
  %add.ptr = getelementptr inbounds i8, ptr %this1, i64 16
  store ptr getelementptr inbounds inrange(-16, 8) ({ [4 x ptr], [3 x ptr] }, ptr @_ZTV1C, i32 0, i32 1, i32 2), ptr %add.ptr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 2), ptr %this1, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define linkonce_odr void @_ZN1BC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV1B, i32 0, i32 0, i32 2), ptr %this1, align 8
  ret void
}

attributes #0 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #2 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 23.0.0git (git@github.com:xiongzile/llvm-project.git c1ef54f3f3c46982dd7dc03c41f66f8e274ac2e6)"}
