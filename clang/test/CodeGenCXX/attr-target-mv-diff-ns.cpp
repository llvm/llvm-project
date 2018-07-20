// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
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

// CHECK: @_Z3fooi.ifunc = ifunc i32 (i32), i32 (i32)* ()* @_Z3fooi.resolver
// CHECK: @_ZN2ns3fooEi.ifunc = ifunc i32 (i32), i32 (i32)* ()* @_ZN2ns3fooEi.resolver

// CHECK: define i32 @_Z3fooi.sse4.2(i32)
// CHECK: ret i32 0
// CHECK: define i32 @_Z3fooi.arch_ivybridge(i32)
// CHECK: ret i32 1
// CHECK: define i32 @_Z3fooi(i32)
// CHECK: ret i32 2

// CHECK: define i32 @_ZN2ns3fooEi.sse4.2(i32)
// CHECK: ret i32 0
// CHECK: define i32 @_ZN2ns3fooEi.arch_ivybridge(i32)
// CHECK: ret i32 1
// CHECK: define i32 @_ZN2ns3fooEi(i32)
// CHECK: ret i32 2

// CHECK: define i32 @_Z3barv()
// CHECK: call i32 @_Z3fooi.ifunc(i32 1)
// CHECK: call i32 @_ZN2ns3fooEi.ifunc(i32 2)

// CHECK: define i32 (i32)* @_Z3fooi.resolver() comdat
// CHECK: ret i32 (i32)* @_Z3fooi.arch_sandybridge
// CHECK: ret i32 (i32)* @_Z3fooi.arch_ivybridge
// CHECK: ret i32 (i32)* @_Z3fooi.sse4.2
// CHECK: ret i32 (i32)* @_Z3fooi
//
// CHECK: define i32 (i32)* @_ZN2ns3fooEi.resolver() comdat
// CHECK: ret i32 (i32)* @_ZN2ns3fooEi.arch_sandybridge
// CHECK: ret i32 (i32)* @_ZN2ns3fooEi.arch_ivybridge
// CHECK: ret i32 (i32)* @_ZN2ns3fooEi.sse4.2
// CHECK: ret i32 (i32)* @_ZN2ns3fooEi

// CHECK: declare i32 @_Z3fooi.arch_sandybridge(i32)
// CHECK: declare i32 @_ZN2ns3fooEi.arch_sandybridge(i32)
