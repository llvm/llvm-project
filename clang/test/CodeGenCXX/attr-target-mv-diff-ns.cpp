// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LINUX
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

// LINUX: @_Z3fooi.ifunc = weak_odr ifunc i32 (i32), i32 (i32)* ()* @_Z3fooi.resolver
// LINUX: @_ZN2ns3fooEi.ifunc = weak_odr ifunc i32 (i32), i32 (i32)* ()* @_ZN2ns3fooEi.resolver

// LINUX: define{{.*}} i32 @_Z3fooi.sse4.2(i32 %0)
// LINUX: ret i32 0
// LINUX: define{{.*}} i32 @_Z3fooi.arch_ivybridge(i32 %0)
// LINUX: ret i32 1
// LINUX: define{{.*}} i32 @_Z3fooi(i32 %0)
// LINUX: ret i32 2

// WINDOWS: define dso_local i32 @"?foo@@YAHH@Z.sse4.2"(i32 %0)
// WINDOWS: ret i32 0
// WINDOWS: define dso_local i32 @"?foo@@YAHH@Z.arch_ivybridge"(i32 %0)
// WINDOWS: ret i32 1
// WINDOWS: define dso_local i32 @"?foo@@YAHH@Z"(i32 %0)
// WINDOWS: ret i32 2

// LINUX: define{{.*}} i32 @_ZN2ns3fooEi.sse4.2(i32 %0)
// LINUX: ret i32 0
// LINUX: define{{.*}} i32 @_ZN2ns3fooEi.arch_ivybridge(i32 %0)
// LINUX: ret i32 1
// LINUX: define{{.*}} i32 @_ZN2ns3fooEi(i32 %0)
// LINUX: ret i32 2

// WINDOWS: define dso_local i32 @"?foo@ns@@YAHH@Z.sse4.2"(i32 %0)
// WINDOWS: ret i32 0
// WINDOWS: define dso_local i32 @"?foo@ns@@YAHH@Z.arch_ivybridge"(i32 %0)
// WINDOWS: ret i32 1
// WINDOWS: define dso_local i32 @"?foo@ns@@YAHH@Z"(i32 %0)
// WINDOWS: ret i32 2

// LINUX: define{{.*}} i32 @_Z3barv()
// LINUX: call i32 @_Z3fooi.ifunc(i32 1)
// LINUX: call i32 @_ZN2ns3fooEi.ifunc(i32 2)

// WINDOWS: define dso_local i32 @"?bar@@YAHXZ"()
// WINDOWS: call i32 @"?foo@@YAHH@Z.resolver"(i32 1)
// WINDOWS: call i32 @"?foo@ns@@YAHH@Z.resolver"(i32 2)

// LINUX: define weak_odr i32 (i32)* @_Z3fooi.resolver() comdat
// LINUX: ret i32 (i32)* @_Z3fooi.arch_sandybridge
// LINUX: ret i32 (i32)* @_Z3fooi.arch_ivybridge
// LINUX: ret i32 (i32)* @_Z3fooi.sse4.2
// LINUX: ret i32 (i32)* @_Z3fooi

// WINDOWS: define weak_odr dso_local i32 @"?foo@@YAHH@Z.resolver"(i32 %0) comdat
// WINDOWS: call i32 @"?foo@@YAHH@Z.arch_sandybridge"(i32 %0)
// WINDOWS: call i32 @"?foo@@YAHH@Z.arch_ivybridge"(i32 %0)
// WINDOWS: call i32 @"?foo@@YAHH@Z.sse4.2"(i32 %0)
// WINDOWS: call i32 @"?foo@@YAHH@Z"(i32 %0)

// LINUX: define weak_odr i32 (i32)* @_ZN2ns3fooEi.resolver() comdat
// LINUX: ret i32 (i32)* @_ZN2ns3fooEi.arch_sandybridge
// LINUX: ret i32 (i32)* @_ZN2ns3fooEi.arch_ivybridge
// LINUX: ret i32 (i32)* @_ZN2ns3fooEi.sse4.2
// LINUX: ret i32 (i32)* @_ZN2ns3fooEi

// WINDOWS: define weak_odr dso_local i32 @"?foo@ns@@YAHH@Z.resolver"(i32 %0) comdat
// WINDOWS: call i32 @"?foo@ns@@YAHH@Z.arch_sandybridge"(i32 %0)
// WINDOWS: call i32 @"?foo@ns@@YAHH@Z.arch_ivybridge"(i32 %0)
// WINDOWS: call i32 @"?foo@ns@@YAHH@Z.sse4.2"(i32 %0)
// WINDOWS: call i32 @"?foo@ns@@YAHH@Z"(i32 %0)

// LINUX: declare i32 @_Z3fooi.arch_sandybridge(i32)
// LINUX: declare i32 @_ZN2ns3fooEi.arch_sandybridge(i32)

// WINDOWS: declare dso_local i32 @"?foo@@YAHH@Z.arch_sandybridge"(i32)
// WINDOWS: declare dso_local i32 @"?foo@ns@@YAHH@Z.arch_sandybridge"(i32)
