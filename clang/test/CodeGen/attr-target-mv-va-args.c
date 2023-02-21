// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LINUX
// RUN: %clang_cc1 -triple x86_64-windows-pc -emit-llvm %s -o - | FileCheck %s --check-prefixes=NO-IFUNC,WINDOWS
// RUN: %clang_cc1 -triple x86_64-fuchsia -emit-llvm %s -o - | FileCheck %s --check-prefixes=NO-IFUNC,FUCHSIA
int __attribute__((target("sse4.2"))) foo(int i, ...) { return 0; }
int __attribute__((target("arch=sandybridge"))) foo(int i, ...);
int __attribute__((target("arch=ivybridge"))) foo(int i, ...) {return 1;}
int __attribute__((target("default"))) foo(int i, ...) { return 2; }

int bar(void) {
  return foo(1, 'a', 1.1) + foo(2, 2.2, "asdf");
}

// LINUX: @foo.ifunc = weak_odr ifunc i32 (i32, ...), ptr @foo.resolver
// LINUX: define{{.*}} i32 @foo.sse4.2(i32 noundef %i, ...)
// LINUX: ret i32 0
// LINUX: define{{.*}} i32 @foo.arch_ivybridge(i32 noundef %i, ...)
// LINUX: ret i32 1
// LINUX: define{{.*}} i32 @foo(i32 noundef %i, ...)
// LINUX: ret i32 2
// LINUX: define{{.*}} i32 @bar()
// LINUX: call i32 (i32, ...) @foo.ifunc(i32 noundef 1, i32 noundef 97, double
// LINUX: call i32 (i32, ...) @foo.ifunc(i32 noundef 2, double noundef 2.2{{[0-9Ee+]+}}, ptr noundef

// LINUX: define weak_odr ptr @foo.resolver() comdat
// LINUX: ret ptr @foo.arch_sandybridge
// LINUX: ret ptr @foo.arch_ivybridge
// LINUX: ret ptr @foo.sse4.2
// LINUX: ret ptr @foo
// LINUX: declare i32 @foo.arch_sandybridge(i32 noundef, ...)

// NO-IFUNC: define dso_local i32 @foo.sse4.2(i32 noundef %i, ...)
// NO-IFUNC: ret i32 0
// NO-IFUNC: define dso_local i32 @foo.arch_ivybridge(i32 noundef %i, ...)
// NO-IFUNC: ret i32 1
// NO-IFUNC: define dso_local i32 @foo(i32 noundef %i, ...)
// NO-IFUNC: ret i32 2
// NO-IFUNC: define dso_local i32 @bar()
// NO-IFUNC: call i32 (i32, ...) @foo.resolver(i32 noundef 1, i32 noundef 97, double
// NO-IFUNC: call i32 (i32, ...) @foo.resolver(i32 noundef 2, double noundef 2.2{{[0-9Ee+]+}}, ptr noundef

// WINDOWS: define weak_odr dso_local i32 @foo.resolver(i32 %0, ...) comdat
// FUCHSIA: define weak_odr i32 @foo.resolver(i32 %0, ...) comdat
// NO-IFUNC: musttail call i32 (i32, ...) @foo.arch_sandybridge
// NO-IFUNC: musttail call i32 (i32, ...) @foo.arch_ivybridge
// NO-IFUNC: musttail call i32 (i32, ...) @foo.sse4.2
// NO-IFUNC: musttail call i32 (i32, ...) @foo
// WINDOWS: declare dso_local i32 @foo.arch_sandybridge(i32 noundef, ...)
// FUCHSIA: declare i32 @foo.arch_sandybridge(i32 noundef, ...)
