// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s


// Each called version should have an IFunc.
// CHECK: @SingleVersion.ifunc = ifunc void (), void ()* ()* @SingleVersion.resolver
// CHECK: @TwoVersions.ifunc = ifunc void (), void ()* ()* @TwoVersions.resolver
// CHECK: @TwoVersionsSameAttr.ifunc = ifunc void (), void ()* ()* @TwoVersionsSameAttr.resolver
// CHECK: @ThreeVersionsSameAttr.ifunc = ifunc void (), void ()* ()* @ThreeVersionsSameAttr.resolver

__attribute__((cpu_specific(ivybridge)))
void SingleVersion(void){}
// CHECK: define void @SingleVersion.S() #[[S:[0-9]+]]

__attribute__((cpu_specific(ivybridge)))
void NotCalled(void){}
// CHECK: define void @NotCalled.S() #[[S]]

// Done before any of the implementations.
__attribute__((cpu_dispatch(ivybridge, knl)))
void TwoVersions(void);
// CHECK: define void ()* @TwoVersions.resolver()
// CHECK: call void @__cpu_indicator_init
// CHECK: ret void ()* @TwoVersions.Z
// CHECK: ret void ()* @TwoVersions.S
// CHECK: call void @llvm.trap
// CHECK: unreachable

__attribute__((cpu_specific(ivybridge)))
void TwoVersions(void){}
// CHECK: define void @TwoVersions.S() #[[S]]

__attribute__((cpu_specific(knl)))
void TwoVersions(void){}
// CHECK: define void @TwoVersions.Z() #[[K:[0-9]+]]

__attribute__((cpu_specific(ivybridge, knl)))
void TwoVersionsSameAttr(void){}
// CHECK: define void @TwoVersionsSameAttr.S() #[[S]]
// CHECK: define void @TwoVersionsSameAttr.Z() #[[K]]

__attribute__((cpu_specific(atom, ivybridge, knl)))
void ThreeVersionsSameAttr(void){}
// CHECK: define void @ThreeVersionsSameAttr.O() #[[O:[0-9]+]]
// CHECK: define void @ThreeVersionsSameAttr.S() #[[S]]
// CHECK: define void @ThreeVersionsSameAttr.Z() #[[K]]

void usages() {
  SingleVersion();
  // CHECK: @SingleVersion.ifunc()
  TwoVersions();
  // CHECK: @TwoVersions.ifunc()
  TwoVersionsSameAttr();
  // CHECK: @TwoVersionsSameAttr.ifunc()
  ThreeVersionsSameAttr();
  // CHECK: @ThreeVersionsSameAttr.ifunc()
}

// has an extra config to emit!
__attribute__((cpu_dispatch(ivybridge, knl, atom)))
void TwoVersionsSameAttr(void);
// CHECK: define void ()* @TwoVersionsSameAttr.resolver()
// CHECK: ret void ()* @TwoVersionsSameAttr.Z
// CHECK: ret void ()* @TwoVersionsSameAttr.S
// CHECK: ret void ()* @TwoVersionsSameAttr.O
// CHECK: call void @llvm.trap
// CHECK: unreachable

__attribute__((cpu_dispatch(atom, ivybridge, knl)))
void ThreeVersionsSameAttr(void){}
// CHECK: define void ()* @ThreeVersionsSameAttr.resolver()
// CHECK: call void @__cpu_indicator_init
// CHECK: ret void ()* @ThreeVersionsSameAttr.Z
// CHECK: ret void ()* @ThreeVersionsSameAttr.S
// CHECK: ret void ()* @ThreeVersionsSameAttr.O
// CHECK: call void @llvm.trap
// CHECK: unreachable

// No Cpu Specific options.
__attribute__((cpu_dispatch(atom, ivybridge, knl)))
void NoSpecifics(void);
// CHECK: define void ()* @NoSpecifics.resolver()
// CHECK: call void @__cpu_indicator_init
// CHECK: ret void ()* @NoSpecifics.Z
// CHECK: ret void ()* @NoSpecifics.S
// CHECK: ret void ()* @NoSpecifics.O
// CHECK: call void @llvm.trap
// CHECK: unreachable

__attribute__((cpu_dispatch(atom, generic, ivybridge, knl)))
void HasGeneric(void);
// CHECK: define void ()* @HasGeneric.resolver()
// CHECK: call void @__cpu_indicator_init
// CHECK: ret void ()* @HasGeneric.Z
// CHECK: ret void ()* @HasGeneric.S
// CHECK: ret void ()* @HasGeneric.O
// CHECK: ret void ()* @HasGeneric.A
// CHECK-NOT: call void @llvm.trap

// CHECK: attributes #[[S]] = {{.*}}"target-features"="+avx,+cmov,+f16c,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
// CHECK: attributes #[[K]] = {{.*}}"target-features"="+adx,+avx,+avx2,+avx512cd,+avx512er,+avx512f,+avx512pf,+bmi,+cmov,+f16c,+fma,+lzcnt,+mmx,+movbe,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave"
// CHECK: attributes #[[O]] = {{.*}}"target-features"="+cmov,+mmx,+movbe,+sse,+sse2,+sse3,+ssse3,+x87"
