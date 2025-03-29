// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=ITANIUM,LINUX
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-macos -emit-llvm %s -o - | FileCheck %s --check-prefixes=ITANIUM,DARWIN
// RUN: %clang_cc1 -std=c++11 -triple x86_64-windows-pc -emit-llvm %s -o - | FileCheck %s --check-prefix=WINDOWS
// Test ensures that this properly differentiates between types in different
// namespaces.
int __attribute__((target("sse4.2"))) foo(int) { return 0; }
int __attribute__((target("arch=sandybridge"))) foo(int);
int __attribute__((target("arch=ivybridge"))) foo(int) {return 1;}
int __attribute__((target("default"))) foo(int) { return 2; }

namespace ns {
int __attribute__((target("sse4.2"))) foo(int) { return 0; }
int __attribute__((target("arch=sandybridge"))) foo(int);
int __attribute__((target("arch=ivybridge"))) foo(int) {return 1;}
int __attribute__((target("default"))) foo(int) { return 2; }
}

int bar() {
  return foo(1) + ns::foo(2);
}

// DARWIN-NOT: comdat

// ITANIUM: @_Z3fooi.ifunc = weak_odr ifunc i32 (i32), ptr @_Z3fooi.resolver
// ITANIUM: @_ZN2ns3fooEi.ifunc = weak_odr ifunc i32 (i32), ptr @_ZN2ns3fooEi.resolver

// ITANIUM: define{{.*}} i32 @_Z3fooi.sse4.2(i32 noundef %0)
// ITANIUM: ret i32 0
// ITANIUM: define{{.*}} i32 @_Z3fooi.arch_ivybridge(i32 noundef %0)
// ITANIUM: ret i32 1
// ITANIUM: define{{.*}} i32 @_Z3fooi(i32 noundef %0)
// ITANIUM: ret i32 2

// WINDOWS: define dso_local noundef i32 @"?foo@@YAHH@Z.sse4.2"(i32 noundef %0)
// WINDOWS: ret i32 0
// WINDOWS: define dso_local noundef i32 @"?foo@@YAHH@Z.arch_ivybridge"(i32 noundef %0)
// WINDOWS: ret i32 1
// WINDOWS: define dso_local noundef i32 @"?foo@@YAHH@Z"(i32 noundef %0)
// WINDOWS: ret i32 2

// ITANIUM: define{{.*}} i32 @_ZN2ns3fooEi.sse4.2(i32 noundef %0)
// ITANIUM: ret i32 0
// ITANIUM: define{{.*}} i32 @_ZN2ns3fooEi.arch_ivybridge(i32 noundef %0)
// ITANIUM: ret i32 1
// ITANIUM: define{{.*}} i32 @_ZN2ns3fooEi(i32 noundef %0)
// ITANIUM: ret i32 2

// WINDOWS: define dso_local noundef i32 @"?foo@ns@@YAHH@Z.sse4.2"(i32 noundef %0)
// WINDOWS: ret i32 0
// WINDOWS: define dso_local noundef i32 @"?foo@ns@@YAHH@Z.arch_ivybridge"(i32 noundef %0)
// WINDOWS: ret i32 1
// WINDOWS: define dso_local noundef i32 @"?foo@ns@@YAHH@Z"(i32 noundef %0)
// WINDOWS: ret i32 2

// ITANIUM: define{{.*}} i32 @_Z3barv()
// ITANIUM: call noundef i32 @_Z3fooi.ifunc(i32 noundef 1)
// ITANIUM: call noundef i32 @_ZN2ns3fooEi.ifunc(i32 noundef 2)

// WINDOWS: define dso_local noundef i32 @"?bar@@YAHXZ"()
// WINDOWS: call noundef i32 @"?foo@@YAHH@Z.resolver"(i32 noundef 1)
// WINDOWS: call noundef i32 @"?foo@ns@@YAHH@Z.resolver"(i32 noundef 2)

// ITANIUM: define weak_odr ptr @_Z3fooi.resolver()
// LINUX-SAME: comdat
// ITANIUM: ret ptr @_Z3fooi.arch_sandybridge
// ITANIUM: ret ptr @_Z3fooi.arch_ivybridge
// ITANIUM: ret ptr @_Z3fooi.sse4.2
// ITANIUM: ret ptr @_Z3fooi

// WINDOWS: define weak_odr dso_local i32 @"?foo@@YAHH@Z.resolver"(i32 %0) comdat
// WINDOWS: call i32 @"?foo@@YAHH@Z.arch_sandybridge"(i32 %0)
// WINDOWS: call i32 @"?foo@@YAHH@Z.arch_ivybridge"(i32 %0)
// WINDOWS: call i32 @"?foo@@YAHH@Z.sse4.2"(i32 %0)
// WINDOWS: call i32 @"?foo@@YAHH@Z"(i32 %0)

// ITANIUM: define weak_odr ptr @_ZN2ns3fooEi.resolver()
// LINUX-SAME: comdat
// ITANIUM: ret ptr @_ZN2ns3fooEi.arch_sandybridge
// ITANIUM: ret ptr @_ZN2ns3fooEi.arch_ivybridge
// ITANIUM: ret ptr @_ZN2ns3fooEi.sse4.2
// ITANIUM: ret ptr @_ZN2ns3fooEi

// WINDOWS: define weak_odr dso_local i32 @"?foo@ns@@YAHH@Z.resolver"(i32 %0) comdat
// WINDOWS: call i32 @"?foo@ns@@YAHH@Z.arch_sandybridge"(i32 %0)
// WINDOWS: call i32 @"?foo@ns@@YAHH@Z.arch_ivybridge"(i32 %0)
// WINDOWS: call i32 @"?foo@ns@@YAHH@Z.sse4.2"(i32 %0)
// WINDOWS: call i32 @"?foo@ns@@YAHH@Z"(i32 %0)

// ITANIUM: declare noundef i32 @_Z3fooi.arch_sandybridge(i32 noundef)
// ITANIUM: declare noundef i32 @_ZN2ns3fooEi.arch_sandybridge(i32 noundef)

// WINDOWS: declare dso_local noundef i32 @"?foo@@YAHH@Z.arch_sandybridge"(i32 noundef)
// WINDOWS: declare dso_local noundef i32 @"?foo@ns@@YAHH@Z.arch_sandybridge"(i32 noundef)
