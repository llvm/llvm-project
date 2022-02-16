// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LINUX
// RUN: %clang_cc1 -triple x86_64-windows-pc -fms-compatibility -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WINDOWS

#ifdef _WIN64
#define ATTR(X) __declspec(X)
#else
#define ATTR(X) __attribute__((X))
#endif // _MSC_VER

// Each version should have an IFunc and an alias.
// LINUX: @TwoVersions = weak_odr alias void (), void ()* @TwoVersions.ifunc
// LINUX: @TwoVersionsSameAttr = weak_odr alias void (), void ()* @TwoVersionsSameAttr.ifunc
// LINUX: @ThreeVersionsSameAttr = weak_odr alias void (), void ()* @ThreeVersionsSameAttr.ifunc
// LINUX: @NoSpecifics = weak_odr alias void (), void ()* @NoSpecifics.ifunc
// LINUX: @HasGeneric = weak_odr alias void (), void ()* @HasGeneric.ifunc
// LINUX: @HasParams = weak_odr alias void (i32, double), void (i32, double)* @HasParams.ifunc
// LINUX: @HasParamsAndReturn = weak_odr alias i32 (i32, double), i32 (i32, double)* @HasParamsAndReturn.ifunc
// LINUX: @GenericAndPentium = weak_odr alias i32 (i32, double), i32 (i32, double)* @GenericAndPentium.ifunc
// LINUX: @DispatchFirst = weak_odr alias i32 (), i32 ()* @DispatchFirst.ifunc

// LINUX: @TwoVersions.ifunc = weak_odr ifunc void (), void ()* ()* @TwoVersions.resolver
// LINUX: @SingleVersion.ifunc = weak_odr ifunc void (), void ()* ()* @SingleVersion.resolver
// LINUX: @TwoVersionsSameAttr.ifunc = weak_odr ifunc void (), void ()* ()* @TwoVersionsSameAttr.resolver
// LINUX: @ThreeVersionsSameAttr.ifunc = weak_odr ifunc void (), void ()* ()* @ThreeVersionsSameAttr.resolver
// LINUX: @NoSpecifics.ifunc = weak_odr ifunc void (), void ()* ()* @NoSpecifics.resolver
// LINUX: @HasGeneric.ifunc = weak_odr ifunc void (), void ()* ()* @HasGeneric.resolver
// LINUX: @HasParams.ifunc = weak_odr ifunc void (i32, double), void (i32, double)* ()* @HasParams.resolver
// LINUX: @HasParamsAndReturn.ifunc = weak_odr ifunc i32 (i32, double), i32 (i32, double)* ()* @HasParamsAndReturn.resolver
// LINUX: @GenericAndPentium.ifunc = weak_odr ifunc i32 (i32, double), i32 (i32, double)* ()* @GenericAndPentium.resolver
// LINUX: @DispatchFirst.ifunc = weak_odr ifunc i32 (), i32 ()* ()* @DispatchFirst.resolver

ATTR(cpu_specific(ivybridge))
void SingleVersion(void){}
// LINUX: define{{.*}} void @SingleVersion.S() #[[S:[0-9]+]]
// WINDOWS: define dso_local void @SingleVersion.S() #[[S:[0-9]+]]

ATTR(cpu_specific(ivybridge))
void NotCalled(void){}
// LINUX: define{{.*}} void @NotCalled.S() #[[S]]
// WINDOWS: define dso_local void @NotCalled.S() #[[S:[0-9]+]]

// Done before any of the implementations.  Also has an undecorated forward
// declaration.
void TwoVersions(void);

ATTR(cpu_dispatch(ivybridge, knl))
void TwoVersions(void);
// LINUX: define weak_odr void ()* @TwoVersions.resolver()
// LINUX: call void @__cpu_indicator_init
// LINUX: ret void ()* @TwoVersions.Z
// LINUX: ret void ()* @TwoVersions.S
// LINUX: call void @llvm.trap
// LINUX: unreachable

// WINDOWS: define weak_odr dso_local void @TwoVersions() comdat
// WINDOWS: call void @__cpu_indicator_init()
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
}

// has an extra config to emit!
ATTR(cpu_dispatch(ivybridge, knl, atom))
void TwoVersionsSameAttr(void);
// LINUX: define weak_odr void ()* @TwoVersionsSameAttr.resolver()
// LINUX: ret void ()* @TwoVersionsSameAttr.Z
// LINUX: ret void ()* @TwoVersionsSameAttr.S
// LINUX: ret void ()* @TwoVersionsSameAttr.O
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
// LINUX: define weak_odr void ()* @ThreeVersionsSameAttr.resolver()
// LINUX: call void @__cpu_indicator_init
// LINUX: ret void ()* @ThreeVersionsSameAttr.Z
// LINUX: ret void ()* @ThreeVersionsSameAttr.S
// LINUX: ret void ()* @ThreeVersionsSameAttr.O
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

// No Cpu Specific options.
ATTR(cpu_dispatch(atom, ivybridge, knl))
void NoSpecifics(void);
// LINUX: define weak_odr void ()* @NoSpecifics.resolver()
// LINUX: call void @__cpu_indicator_init
// LINUX: ret void ()* @NoSpecifics.Z
// LINUX: ret void ()* @NoSpecifics.S
// LINUX: ret void ()* @NoSpecifics.O
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
// LINUX: define weak_odr void ()* @HasGeneric.resolver()
// LINUX: call void @__cpu_indicator_init
// LINUX: ret void ()* @HasGeneric.Z
// LINUX: ret void ()* @HasGeneric.S
// LINUX: ret void ()* @HasGeneric.O
// LINUX: ret void ()* @HasGeneric.A
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
// LINUX: define weak_odr void (i32, double)* @HasParams.resolver()
// LINUX: call void @__cpu_indicator_init
// LINUX: ret void (i32, double)* @HasParams.Z
// LINUX: ret void (i32, double)* @HasParams.S
// LINUX: ret void (i32, double)* @HasParams.O
// LINUX: ret void (i32, double)* @HasParams.A
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
// LINUX: define weak_odr i32 (i32, double)* @HasParamsAndReturn.resolver()
// LINUX: call void @__cpu_indicator_init
// LINUX: ret i32 (i32, double)* @HasParamsAndReturn.Z
// LINUX: ret i32 (i32, double)* @HasParamsAndReturn.S
// LINUX: ret i32 (i32, double)* @HasParamsAndReturn.O
// LINUX: ret i32 (i32, double)* @HasParamsAndReturn.A
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
// LINUX: define weak_odr i32 (i32, double)* @GenericAndPentium.resolver()
// LINUX: call void @__cpu_indicator_init
// LINUX: ret i32 (i32, double)* @GenericAndPentium.O
// LINUX: ret i32 (i32, double)* @GenericAndPentium.B
// LINUX-NOT: ret i32 (i32, double)* @GenericAndPentium.A
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
// LINUX: define weak_odr i32 ()* @DispatchFirst.resolver
// LINUX: ret i32 ()* @DispatchFirst.O
// LINUX: ret i32 ()* @DispatchFirst.B

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

// CHECK: attributes #[[S]] = {{.*}}"target-features"="+avx,+cmov,+crc32,+cx8,+f16c,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
// CHECK: attributes #[[K]] = {{.*}}"target-features"="+adx,+avx,+avx2,+avx512cd,+avx512er,+avx512f,+avx512pf,+bmi,+cmov,+crc32,+cx8,+f16c,+fma,+lzcnt,+mmx,+movbe,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
// CHECK: attributes #[[O]] = {{.*}}"target-features"="+cmov,+cx8,+mmx,+movbe,+sse,+sse2,+sse3,+ssse3,+x87"
