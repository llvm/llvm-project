// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LINUX
// RUN: %clang_cc1 -triple x86_64-windows-pc -fms-compatibility -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WINDOWS

#ifdef _WIN64
#define ATTR(X) __declspec(X)
#else
#define ATTR(X) __attribute__((X))
#endif // _WIN64

// Each version should have an IFunc and an alias.
// LINUX: @SingleVersion = weak_odr alias void (), ptr @SingleVersion.ifunc
// LINUX: @TwoVersions = weak_odr alias void (), ptr @TwoVersions.ifunc
// LINUX: @OrderDispatchUsageSpecific = weak_odr alias void (), ptr @OrderDispatchUsageSpecific.ifunc
// LINUX: @TwoVersionsSameAttr = weak_odr alias void (), ptr @TwoVersionsSameAttr.ifunc
// LINUX: @ThreeVersionsSameAttr = weak_odr alias void (), ptr @ThreeVersionsSameAttr.ifunc
// LINUX: @OrderSpecificUsageDispatch = weak_odr alias void (), ptr @OrderSpecificUsageDispatch.ifunc
// LINUX: @NoSpecifics = weak_odr alias void (), ptr @NoSpecifics.ifunc
// LINUX: @HasGeneric = weak_odr alias void (), ptr @HasGeneric.ifunc
// LINUX: @HasParams = weak_odr alias void (i32, double), ptr @HasParams.ifunc
// LINUX: @HasParamsAndReturn = weak_odr alias i32 (i32, double), ptr @HasParamsAndReturn.ifunc
// LINUX: @GenericAndPentium = weak_odr alias i32 (i32, double), ptr @GenericAndPentium.ifunc
// LINUX: @DispatchFirst = weak_odr alias i32 (), ptr @DispatchFirst.ifunc

// LINUX: @SingleVersion.ifunc = weak_odr ifunc void (), ptr @SingleVersion.resolver
// LINUX: @TwoVersions.ifunc = weak_odr ifunc void (), ptr @TwoVersions.resolver
// LINUX: @OrderDispatchUsageSpecific.ifunc = weak_odr ifunc void (), ptr @OrderDispatchUsageSpecific.resolver
// LINUX: @TwoVersionsSameAttr.ifunc = weak_odr ifunc void (), ptr @TwoVersionsSameAttr.resolver
// LINUX: @ThreeVersionsSameAttr.ifunc = weak_odr ifunc void (), ptr @ThreeVersionsSameAttr.resolver
// LINUX: @OrderSpecificUsageDispatch.ifunc = weak_odr ifunc void (), ptr @OrderSpecificUsageDispatch.resolver
// LINUX: @NoSpecifics.ifunc = weak_odr ifunc void (), ptr @NoSpecifics.resolver
// LINUX: @HasGeneric.ifunc = weak_odr ifunc void (), ptr @HasGeneric.resolver
// LINUX: @HasParams.ifunc = weak_odr ifunc void (i32, double), ptr @HasParams.resolver
// LINUX: @HasParamsAndReturn.ifunc = weak_odr ifunc i32 (i32, double), ptr @HasParamsAndReturn.resolver
// LINUX: @GenericAndPentium.ifunc = weak_odr ifunc i32 (i32, double), ptr @GenericAndPentium.resolver
// LINUX: @DispatchFirst.ifunc = weak_odr ifunc i32 (), ptr @DispatchFirst.resolver

ATTR(cpu_specific(ivybridge))
void SingleVersion(void){}
// LINUX: define{{.*}} void @SingleVersion.S() #[[S:[0-9]+]]
// WINDOWS: define dso_local void @SingleVersion.S() #[[S:[0-9]+]]

ATTR(cpu_dispatch(ivybridge))
void SingleVersion(void);
// LINUX: define weak_odr ptr @SingleVersion.resolver()
// LINUX: call void @__cpu_indicator_init
// LINUX: %[[FEAT_INIT:.+]] = load i32, ptr getelementptr inbounds ({ i32, i32, i32, [1 x i32] }, ptr @__cpu_model, i32 0, i32 3, i32 0), align 4
// LINUX: %[[FEAT_JOIN:.+]] = and i32 %[[FEAT_INIT]], 525311
// LINUX: %[[FEAT_CHECK:.+]] = icmp eq i32 %[[FEAT_JOIN]], 525311
// LINUX: ret ptr @SingleVersion.S
// LINUX: call void @llvm.trap
// LINUX: unreachable

// WINDOWS: define weak_odr dso_local void @SingleVersion() comdat
// WINDOWS: call void @__cpu_indicator_init()
// WINDOWS: %[[FEAT_INIT:.+]] = load i32, ptr getelementptr inbounds ({ i32, i32, i32, [1 x i32] }, ptr @__cpu_model, i32 0, i32 3, i32 0), align 4
// WINDOWS: %[[FEAT_JOIN:.+]] = and i32 %[[FEAT_INIT]], 525311
// WINDOWS: %[[FEAT_CHECK:.+]] = icmp eq i32 %[[FEAT_JOIN]], 525311
// WINDOWS: call void @SingleVersion.S()
// WINDOWS-NEXT: ret void
// WINDOWS: call void @llvm.trap
// WINDOWS: unreachable

ATTR(cpu_specific(ivybridge))
void NotCalled(void){}
// LINUX: define{{.*}} void @NotCalled.S() #[[S]]
// WINDOWS: define dso_local void @NotCalled.S() #[[S:[0-9]+]]

// Done before any of the implementations.  Also has an undecorated forward
// declaration.
void TwoVersions(void);

ATTR(cpu_dispatch(ivybridge, knl))
void TwoVersions(void);
// LINUX: define weak_odr ptr @TwoVersions.resolver()
// LINUX: call void @__cpu_indicator_init
// LINUX: %[[FEAT_INIT:.+]] = load i32, ptr getelementptr inbounds ({ i32, i32, i32, [1 x i32] }, ptr @__cpu_model, i32 0, i32 3, i32 0), align 4
// LINUX: %[[FEAT_JOIN:.+]] = and i32 %[[FEAT_INIT]], 59754495
// LINUX: %[[FEAT_CHECK:.+]] = icmp eq i32 %[[FEAT_JOIN]], 59754495
// LINUX: ret ptr @TwoVersions.Z
// LINUX: ret ptr @TwoVersions.S
// LINUX: call void @llvm.trap
// LINUX: unreachable

// WINDOWS: define weak_odr dso_local void @TwoVersions() comdat
// WINDOWS: call void @__cpu_indicator_init()
// WINDOWS: %[[FEAT_INIT:.+]] = load i32, ptr getelementptr inbounds ({ i32, i32, i32, [1 x i32] }, ptr @__cpu_model, i32 0, i32 3, i32 0), align 4
// WINDOWS: %[[FEAT_JOIN:.+]] = and i32 %[[FEAT_INIT]], 59754495
// WINDOWS: %[[FEAT_CHECK:.+]] = icmp eq i32 %[[FEAT_JOIN]], 59754495
// WINDOWS: call void @TwoVersions.Z()
// WINDOWS-NEXT: ret void
// WINDOWS: call void @TwoVersions.S()
// WINDOWS-NEXT: ret void
// WINDOWS: call void @llvm.trap
// WINDOWS: unreachable

ATTR(cpu_specific(ivybridge))
void TwoVersions(void){}
// CHECK: define {{.*}}void @TwoVersions.S() #[[S]]

ATTR(cpu_specific(knl))
void TwoVersions(void){}
// CHECK: define {{.*}}void @TwoVersions.Z() #[[K:[0-9]+]]

ATTR(cpu_specific(ivybridge, knl))
void TwoVersionsSameAttr(void){}
// CHECK: define {{.*}}void @TwoVersionsSameAttr.S() #[[S]]
// CHECK: define {{.*}}void @TwoVersionsSameAttr.Z() #[[K]]

ATTR(cpu_specific(atom, ivybridge, knl))
void ThreeVersionsSameAttr(void){}
// CHECK: define {{.*}}void @ThreeVersionsSameAttr.O() #[[O:[0-9]+]]
// CHECK: define {{.*}}void @ThreeVersionsSameAttr.S() #[[S]]
// CHECK: define {{.*}}void @ThreeVersionsSameAttr.Z() #[[K]]

ATTR(cpu_specific(knl))
void CpuSpecificNoDispatch(void) {}
// CHECK: define {{.*}}void @CpuSpecificNoDispatch.Z() #[[K:[0-9]+]]

ATTR(cpu_dispatch(knl))
void OrderDispatchUsageSpecific(void);
// LINUX: define weak_odr ptr @OrderDispatchUsageSpecific.resolver()
// LINUX: call void @__cpu_indicator_init
// LINUX: ret ptr @OrderDispatchUsageSpecific.Z
// LINUX: call void @llvm.trap
// LINUX: unreachable

// WINDOWS: define weak_odr dso_local void @OrderDispatchUsageSpecific() comdat
// WINDOWS: call void @__cpu_indicator_init()
// WINDOWS: call void @OrderDispatchUsageSpecific.Z()
// WINDOWS-NEXT: ret void
// WINDOWS: call void @llvm.trap
// WINDOWS: unreachable

// CHECK: define {{.*}}void @OrderDispatchUsageSpecific.Z()

ATTR(cpu_specific(knl))
void OrderSpecificUsageDispatch(void) {}
// CHECK: define {{.*}}void @OrderSpecificUsageDispatch.Z() #[[K:[0-9]+]]

void usages(void) {
  SingleVersion();
  // LINUX: @SingleVersion.ifunc()
  // WINDOWS: @SingleVersion()
  TwoVersions();
  // LINUX: @TwoVersions.ifunc()
  // WINDOWS: @TwoVersions()
  TwoVersionsSameAttr();
  // LINUX: @TwoVersionsSameAttr.ifunc()
  // WINDOWS: @TwoVersionsSameAttr()
  ThreeVersionsSameAttr();
  // LINUX: @ThreeVersionsSameAttr.ifunc()
  // WINDOWS: @ThreeVersionsSameAttr()
  CpuSpecificNoDispatch();
  // LINUX: @CpuSpecificNoDispatch.ifunc()
  // WINDOWS: @CpuSpecificNoDispatch()
  OrderDispatchUsageSpecific();
  // LINUX: @OrderDispatchUsageSpecific.ifunc()
  // WINDOWS: @OrderDispatchUsageSpecific()
  OrderSpecificUsageDispatch();
  // LINUX: @OrderSpecificUsageDispatch.ifunc()
  // WINDOWS: @OrderSpecificUsageDispatch()
}

// LINUX: declare void @CpuSpecificNoDispatch.ifunc()

// has an extra config to emit!
ATTR(cpu_dispatch(ivybridge, knl, atom))
void TwoVersionsSameAttr(void);
// LINUX: define weak_odr ptr @TwoVersionsSameAttr.resolver()
// LINUX: ret ptr @TwoVersionsSameAttr.Z
// LINUX: ret ptr @TwoVersionsSameAttr.S
// LINUX: ret ptr @TwoVersionsSameAttr.O
// LINUX: call void @llvm.trap
// LINUX: unreachable

// WINDOWS: define weak_odr dso_local void @TwoVersionsSameAttr() comdat
// WINDOWS: call void @TwoVersionsSameAttr.Z
// WINDOWS-NEXT: ret void
// WINDOWS: call void @TwoVersionsSameAttr.S
// WINDOWS-NEXT: ret void
// WINDOWS: call void @TwoVersionsSameAttr.O
// WINDOWS-NEXT: ret void
// WINDOWS: call void @llvm.trap
// WINDOWS: unreachable

ATTR(cpu_dispatch(atom, ivybridge, knl))
void ThreeVersionsSameAttr(void){}
// LINUX: define weak_odr ptr @ThreeVersionsSameAttr.resolver()
// LINUX: call void @__cpu_indicator_init
// LINUX: ret ptr @ThreeVersionsSameAttr.Z
// LINUX: ret ptr @ThreeVersionsSameAttr.S
// LINUX: ret ptr @ThreeVersionsSameAttr.O
// LINUX: call void @llvm.trap
// LINUX: unreachable

// WINDOWS: define weak_odr dso_local void @ThreeVersionsSameAttr() comdat
// WINDOWS: call void @__cpu_indicator_init
// WINDOWS: call void @ThreeVersionsSameAttr.Z
// WINDOWS-NEXT: ret void
// WINDOWS: call void @ThreeVersionsSameAttr.S
// WINDOWS-NEXT: ret void
// WINDOWS: call void @ThreeVersionsSameAttr.O
// WINDOWS-NEXT: ret void
// WINDOWS: call void @llvm.trap
// WINDOWS: unreachable

ATTR(cpu_dispatch(knl))
void OrderSpecificUsageDispatch(void);
// LINUX: define weak_odr ptr @OrderSpecificUsageDispatch.resolver()
// LINUX: ret ptr @OrderSpecificUsageDispatch.Z

// WINDOWS: define weak_odr dso_local void @OrderSpecificUsageDispatch() comdat
// WINDOWS: call void @__cpu_indicator_init
// WINDOWS: call void @OrderSpecificUsageDispatch.Z
// WINDOWS-NEXT: ret void

// No Cpu Specific options.
ATTR(cpu_dispatch(atom, ivybridge, knl))
void NoSpecifics(void);
// LINUX: define weak_odr ptr @NoSpecifics.resolver()
// LINUX: call void @__cpu_indicator_init
// LINUX: ret ptr @NoSpecifics.Z
// LINUX: ret ptr @NoSpecifics.S
// LINUX: ret ptr @NoSpecifics.O
// LINUX: call void @llvm.trap
// LINUX: unreachable

// WINDOWS: define weak_odr dso_local void @NoSpecifics() comdat
// WINDOWS: call void @__cpu_indicator_init
// WINDOWS: call void @NoSpecifics.Z
// WINDOWS-NEXT: ret void
// WINDOWS: call void @NoSpecifics.S
// WINDOWS-NEXT: ret void
// WINDOWS: call void @NoSpecifics.O
// WINDOWS-NEXT: ret void
// WINDOWS: call void @llvm.trap
// WINDOWS: unreachable

ATTR(cpu_dispatch(atom, generic, ivybridge, knl))
void HasGeneric(void);
// LINUX: define weak_odr ptr @HasGeneric.resolver()
// LINUX: call void @__cpu_indicator_init
// LINUX: ret ptr @HasGeneric.Z
// LINUX: ret ptr @HasGeneric.S
// LINUX: ret ptr @HasGeneric.O
// LINUX: ret ptr @HasGeneric.A
// LINUX-NOT: call void @llvm.trap

// WINDOWS: define weak_odr dso_local void @HasGeneric() comdat
// WINDOWS: call void @__cpu_indicator_init
// WINDOWS: call void @HasGeneric.Z
// WINDOWS-NEXT: ret void
// WINDOWS: call void @HasGeneric.S
// WINDOWS-NEXT: ret void
// WINDOWS: call void @HasGeneric.O
// WINDOWS-NEXT: ret void
// WINDOWS: call void @HasGeneric.A
// WINDOWS-NEXT: ret void
// WINDOWS-NOT: call void @llvm.trap

ATTR(cpu_dispatch(atom, generic, ivybridge, knl))
void HasParams(int i, double d);
// LINUX: define weak_odr ptr @HasParams.resolver()
// LINUX: call void @__cpu_indicator_init
// LINUX: ret ptr @HasParams.Z
// LINUX: ret ptr @HasParams.S
// LINUX: ret ptr @HasParams.O
// LINUX: ret ptr @HasParams.A
// LINUX-NOT: call void @llvm.trap

// WINDOWS: define weak_odr dso_local void @HasParams(i32 %0, double %1) comdat
// WINDOWS: call void @__cpu_indicator_init
// WINDOWS: call void @HasParams.Z(i32 %0, double %1)
// WINDOWS-NEXT: ret void
// WINDOWS: call void @HasParams.S(i32 %0, double %1)
// WINDOWS-NEXT: ret void
// WINDOWS: call void @HasParams.O(i32 %0, double %1)
// WINDOWS-NEXT: ret void
// WINDOWS: call void @HasParams.A(i32 %0, double %1)
// WINDOWS-NEXT: ret void
// WINDOWS-NOT: call void @llvm.trap

ATTR(cpu_dispatch(atom, generic, ivybridge, knl))
int HasParamsAndReturn(int i, double d);
// LINUX: define weak_odr ptr @HasParamsAndReturn.resolver()
// LINUX: call void @__cpu_indicator_init
// LINUX: ret ptr @HasParamsAndReturn.Z
// LINUX: ret ptr @HasParamsAndReturn.S
// LINUX: ret ptr @HasParamsAndReturn.O
// LINUX: ret ptr @HasParamsAndReturn.A
// LINUX-NOT: call void @llvm.trap

// WINDOWS: define weak_odr dso_local i32 @HasParamsAndReturn(i32 %0, double %1) comdat
// WINDOWS: call void @__cpu_indicator_init
// WINDOWS: %[[RET:.+]] = musttail call i32 @HasParamsAndReturn.Z(i32 %0, double %1)
// WINDOWS-NEXT: ret i32 %[[RET]]
// WINDOWS: %[[RET:.+]] = musttail call i32 @HasParamsAndReturn.S(i32 %0, double %1)
// WINDOWS-NEXT: ret i32 %[[RET]]
// WINDOWS: %[[RET:.+]] = musttail call i32 @HasParamsAndReturn.O(i32 %0, double %1)
// WINDOWS-NEXT: ret i32 %[[RET]]
// WINDOWS: %[[RET:.+]] = musttail call i32 @HasParamsAndReturn.A(i32 %0, double %1)
// WINDOWS-NEXT: ret i32 %[[RET]]
// WINDOWS-NOT: call void @llvm.trap

ATTR(cpu_dispatch(atom, generic, pentium))
int GenericAndPentium(int i, double d);
// LINUX: define weak_odr ptr @GenericAndPentium.resolver()
// LINUX: call void @__cpu_indicator_init
// LINUX: ret ptr @GenericAndPentium.O
// LINUX: ret ptr @GenericAndPentium.B
// LINUX-NOT: ret ptr @GenericAndPentium.A
// LINUX-NOT: call void @llvm.trap

// WINDOWS: define weak_odr dso_local i32 @GenericAndPentium(i32 %0, double %1) comdat
// WINDOWS: call void @__cpu_indicator_init
// WINDOWS: %[[RET:.+]] = musttail call i32 @GenericAndPentium.O(i32 %0, double %1)
// WINDOWS-NEXT: ret i32 %[[RET]]
// WINDOWS: %[[RET:.+]] = musttail call i32 @GenericAndPentium.B(i32 %0, double %1)
// WINDOWS-NEXT: ret i32 %[[RET]]
// WINDOWS-NOT: call i32 @GenericAndPentium.A
// WINDOWS-NOT: call void @llvm.trap

ATTR(cpu_dispatch(atom, pentium))
int DispatchFirst(void);
// LINUX: define weak_odr ptr @DispatchFirst.resolver
// LINUX: ret ptr @DispatchFirst.O
// LINUX: ret ptr @DispatchFirst.B

// WINDOWS: define weak_odr dso_local i32 @DispatchFirst() comdat
// WINDOWS: %[[RET:.+]] = musttail call i32 @DispatchFirst.O()
// WINDOWS-NEXT: ret i32 %[[RET]]
// WINDOWS: %[[RET:.+]] = musttail call i32 @DispatchFirst.B()
// WINDOWS-NEXT: ret i32 %[[RET]]

ATTR(cpu_specific(atom))
int DispatchFirst(void) {return 0;}
// LINUX: define{{.*}} i32 @DispatchFirst.O
// LINUX: ret i32 0

// WINDOWS: define dso_local i32 @DispatchFirst.O()
// WINDOWS: ret i32 0

ATTR(cpu_specific(pentium))
int DispatchFirst(void) {return 1;}
// LINUX: define{{.*}} i32 @DispatchFirst.B
// LINUX: ret i32 1

// WINDOWS: define dso_local i32 @DispatchFirst.B
// WINDOWS: ret i32 1

ATTR(cpu_specific(knl))
void OrderDispatchUsageSpecific(void) {}

// CHECK: attributes #[[S]] = {{.*}}"target-features"="+avx,+cmov,+crc32,+cx16,+cx8,+f16c,+fsgsbase,+fxsr,+mmx,+pclmul,+popcnt,+rdrnd,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt"
// CHECK-SAME: "tune-cpu"="ivybridge"
// CHECK: attributes #[[K]] = {{.*}}"target-features"="+adx,+aes,+avx,+avx2,+avx512cd,+avx512er,+avx512f,+avx512pf,+bmi,+bmi2,+cmov,+crc32,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+prefetchwt1,+prfchw,+rdrnd,+rdseed,+sahf,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsaveopt"
// CHECK-SAME: "tune-cpu"="knl"
// CHECK: attributes #[[O]] = {{.*}}"target-features"="+cmov,+cx16,+cx8,+fxsr,+mmx,+movbe,+sahf,+sse,+sse2,+sse3,+ssse3,+x87"
// CHECK-SAME: "tune-cpu"="atom"
