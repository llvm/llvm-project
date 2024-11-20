// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=ITANIUM,LINUX
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-macos -emit-llvm %s -o - | FileCheck %s --check-prefixes=ITANIUM,DARWIN
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

// DARWIN-NOT: comdat

// ITANIUM: @_ZN1S3fooEi.ifunc = weak_odr ifunc i32 (ptr, i32), ptr @_ZN1S3fooEi.resolver

// ITANIUM: define{{.*}} i32 @_ZN1S3fooEi(ptr {{[^,]*}} %this, i32 noundef %0)
// ITANIUM: ret i32 2

// WINDOWS: define dso_local noundef i32 @"?foo@S@@QEAAHH@Z"(ptr {{[^,]*}} %this, i32 noundef %0)
// WINDOWS: ret i32 2

// ITANIUM: define{{.*}} i32 @_ZN1S3fooEi.sse4.2(ptr {{[^,]*}} %this, i32 noundef %0)
// ITANIUM: ret i32 0

// WINDOWS: define dso_local noundef i32 @"?foo@S@@QEAAHH@Z.sse4.2"(ptr {{[^,]*}} %this, i32 noundef %0)
// WINDOWS: ret i32 0

// ITANIUM: define{{.*}} i32 @_ZN1S3fooEi.arch_ivybridge(ptr {{[^,]*}} %this, i32 noundef %0)
// ITANIUM: ret i32 1

// WINDOWS: define dso_local noundef i32 @"?foo@S@@QEAAHH@Z.arch_ivybridge"(ptr {{[^,]*}} %this, i32 noundef %0)
// WINDOWS: ret i32 1

// ITANIUM: define{{.*}} i32 @_Z3barv()
// ITANIUM: %s = alloca %struct.S, align 1
// ITANIUM: %call = call noundef i32 @_ZN1S3fooEi.ifunc(ptr {{[^,]*}} %s, i32 noundef 0)

// WINDOWS: define dso_local noundef i32 @"?bar@@YAHXZ"()
// WINDOWS: %s = alloca %struct.S, align 1
// WINDOWS: %call = call noundef i32 @"?foo@S@@QEAAHH@Z.resolver"(ptr {{[^,]*}} %s, i32 noundef 0)

// ITANIUM: define weak_odr ptr @_ZN1S3fooEi.resolver()
// LINUX-SAME: comdat
// ITANIUM: ret ptr @_ZN1S3fooEi.arch_sandybridge
// ITANIUM: ret ptr @_ZN1S3fooEi.arch_ivybridge
// ITANIUM: ret ptr @_ZN1S3fooEi.sse4.2
// ITANIUM: ret ptr @_ZN1S3fooEi

// WINDOWS: define weak_odr dso_local i32 @"?foo@S@@QEAAHH@Z.resolver"(ptr %0, i32 %1) comdat
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z.arch_sandybridge"(ptr %0, i32 %1)
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z.arch_ivybridge"(ptr %0, i32 %1)
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z.sse4.2"(ptr %0, i32 %1)
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z"(ptr %0, i32 %1)

// ITANIUM: declare noundef i32 @_ZN1S3fooEi.arch_sandybridge(ptr {{[^,]*}}, i32 noundef)

// WINDOWS: declare dso_local noundef i32 @"?foo@S@@QEAAHH@Z.arch_sandybridge"(ptr {{[^,]*}}, i32 noundef)
