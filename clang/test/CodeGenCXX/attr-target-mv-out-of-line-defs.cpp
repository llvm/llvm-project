// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LINUX
// RUN: %clang_cc1 -std=c++11 -triple x86_64-windows-pc -emit-llvm %s -o - | FileCheck %s --check-prefix=WINDOWS
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

// LINUX: @_ZN1S3fooEi.ifunc = ifunc i32 (%struct.S*, i32), i32 (%struct.S*, i32)* ()* @_ZN1S3fooEi.resolver

// LINUX: define i32 @_ZN1S3fooEi(%struct.S* %this, i32)
// LINUX: ret i32 2

// WINDOWS: define dso_local i32 @"?foo@S@@QEAAHH@Z"(%struct.S* %this, i32)
// WINDOWS: ret i32 2

// LINUX: define i32 @_ZN1S3fooEi.sse4.2(%struct.S* %this, i32)
// LINUX: ret i32 0

// WINDOWS: define dso_local i32 @"?foo@S@@QEAAHH@Z.sse4.2"(%struct.S* %this, i32)
// WINDOWS: ret i32 0

// LINUX: define i32 @_ZN1S3fooEi.arch_ivybridge(%struct.S* %this, i32)
// LINUX: ret i32 1

// WINDOWS: define dso_local i32 @"?foo@S@@QEAAHH@Z.arch_ivybridge"(%struct.S* %this, i32)
// WINDOWS: ret i32 1

// LINUX: define i32 @_Z3barv()
// LINUX: %s = alloca %struct.S, align 1
// LINUX: %call = call i32 @_ZN1S3fooEi.ifunc(%struct.S* %s, i32 0)

// WINDOWS: define dso_local i32 @"?bar@@YAHXZ"()
// WINDOWS: %s = alloca %struct.S, align 1
// WINDOWS: %call = call i32 @"?foo@S@@QEAAHH@Z.resolver"(%struct.S* %s, i32 0)

// LINUX: define i32 (%struct.S*, i32)* @_ZN1S3fooEi.resolver() comdat
// LINUX: ret i32 (%struct.S*, i32)* @_ZN1S3fooEi.arch_sandybridge
// LINUX: ret i32 (%struct.S*, i32)* @_ZN1S3fooEi.arch_ivybridge
// LINUX: ret i32 (%struct.S*, i32)* @_ZN1S3fooEi.sse4.2
// LINUX: ret i32 (%struct.S*, i32)* @_ZN1S3fooEi

// WINDOWS: define dso_local i32 @"?foo@S@@QEAAHH@Z.resolver"(%struct.S*, i32) comdat
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z.arch_sandybridge"(%struct.S* %0, i32 %1)
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z.arch_ivybridge"(%struct.S* %0, i32 %1)
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z.sse4.2"(%struct.S* %0, i32 %1)
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z"(%struct.S* %0, i32 %1)

// LINUX: declare i32 @_ZN1S3fooEi.arch_sandybridge(%struct.S*, i32)

// WINDOWS: declare dso_local i32 @"?foo@S@@QEAAHH@Z.arch_sandybridge"(%struct.S*, i32)
