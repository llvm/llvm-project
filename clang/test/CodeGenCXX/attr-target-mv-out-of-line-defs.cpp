// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
struct S {
  int __attribute__((target("sse4.2"))) foo(int);
  int __attribute__((target("arch=sandybridge"))) foo(int);
  int __attribute__((target("arch=ivybridge"))) foo(int);
  int __attribute__((target("default"))) foo(int);
};

int __attribute__((target("default"))) S::foo(int) { return 2; }
int __attribute__((target("sse4.2"))) S::foo(int) { return 0; }
int __attribute__((target("arch=ivybridge"))) S::foo(int) { return 1; }

int bar() {
  S s;
  return s.foo(0);
}

// CHECK: @_ZN1S3fooEi.ifunc = ifunc i32 (%struct.S*, i32), i32 (%struct.S*, i32)* ()* @_ZN1S3fooEi.resolver

// CHECK: define i32 @_ZN1S3fooEi(%struct.S* %this, i32)
// CHECK: ret i32 2

// CHECK: define i32 @_ZN1S3fooEi.sse4.2(%struct.S* %this, i32)
// CHECK: ret i32 0

// CHECK: define i32 @_ZN1S3fooEi.arch_ivybridge(%struct.S* %this, i32)
// CHECK: ret i32 1

// CHECK: define i32 @_Z3barv()
// CHECK: %s = alloca %struct.S, align 1
// CHECK: %call = call i32 @_ZN1S3fooEi.ifunc(%struct.S* %s, i32 0)

// CHECK: define i32 (%struct.S*, i32)* @_ZN1S3fooEi.resolver() comdat
// CHECK: ret i32 (%struct.S*, i32)* @_ZN1S3fooEi.arch_sandybridge
// CHECK: ret i32 (%struct.S*, i32)* @_ZN1S3fooEi.arch_ivybridge
// CHECK: ret i32 (%struct.S*, i32)* @_ZN1S3fooEi.sse4.2
// CHECK: ret i32 (%struct.S*, i32)* @_ZN1S3fooEi

// CHECK: declare i32 @_ZN1S3fooEi.arch_sandybridge(%struct.S*, i32)
