// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

int __attribute__((target("sse4.2"))) foo_overload(int) { return 0; }
int __attribute__((target("arch=sandybridge"))) foo_overload(int);
int __attribute__((target("arch=ivybridge"))) foo_overload(int) {return 1;}
int __attribute__((target("default"))) foo_overload(int) { return 2; }
int __attribute__((target("sse4.2"))) foo_overload(void) { return 0; }
int __attribute__((target("arch=sandybridge"))) foo_overload(void);
int __attribute__((target("arch=ivybridge"))) foo_overload(void) {return 1;}
int __attribute__((target("default"))) foo_overload(void) { return 2; }

int bar2() {
  return foo_overload() + foo_overload(1);
}

// CHECK: @_Z12foo_overloadv.ifunc = ifunc i32 (), i32 ()* ()* @_Z12foo_overloadv.resolver
// CHECK: @_Z12foo_overloadi.ifunc = ifunc i32 (i32), i32 (i32)* ()* @_Z12foo_overloadi.resolver


// CHECK: define i32 @_Z12foo_overloadi.sse4.2(i32)
// CHECK: ret i32 0
// CHECK: define i32 @_Z12foo_overloadi.arch_ivybridge(i32)
// CHECK: ret i32 1
// CHECK: define i32 @_Z12foo_overloadi(i32)
// CHECK: ret i32 2
// CHECK: define i32 @_Z12foo_overloadv.sse4.2()
// CHECK: ret i32 0
// CHECK: define i32 @_Z12foo_overloadv.arch_ivybridge()
// CHECK: ret i32 1
// CHECK: define i32 @_Z12foo_overloadv()
// CHECK: ret i32 2

// CHECK: define i32 @_Z4bar2v()
// CHECK: call i32 @_Z12foo_overloadv.ifunc()
// CHECK: call i32 @_Z12foo_overloadi.ifunc(i32 1)

// CHECK: define i32 ()* @_Z12foo_overloadv.resolver() comdat
// CHECK: ret i32 ()* @_Z12foo_overloadv.arch_sandybridge
// CHECK: ret i32 ()* @_Z12foo_overloadv.arch_ivybridge
// CHECK: ret i32 ()* @_Z12foo_overloadv.sse4.2
// CHECK: ret i32 ()* @_Z12foo_overloadv

// CHECK: define i32 (i32)* @_Z12foo_overloadi.resolver() comdat
// CHECK: ret i32 (i32)* @_Z12foo_overloadi.arch_sandybridge
// CHECK: ret i32 (i32)* @_Z12foo_overloadi.arch_ivybridge
// CHECK: ret i32 (i32)* @_Z12foo_overloadi.sse4.2
// CHECK: ret i32 (i32)* @_Z12foo_overloadi

// CHECK: declare i32 @_Z12foo_overloadv.arch_sandybridge()
// CHECK: declare i32 @_Z12foo_overloadi.arch_sandybridge(i32)
