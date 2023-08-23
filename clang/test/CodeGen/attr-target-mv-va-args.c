// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=IFUNC-ELF
// RUN: %clang_cc1 -triple x86_64-pc-freebsd -emit-llvm %s -o - | FileCheck %s --check-prefix=IFUNC-ELF
// RUN: %clang_cc1 -triple x86_64-windows-pc -emit-llvm %s -o - | FileCheck %s --check-prefixes=NO-IFUNC,WINDOWS
// RUN: %clang_cc1 -triple x86_64-linux-musl -emit-llvm %s -o - | FileCheck %s --check-prefixes=NO-IFUNC,NO-IFUNC-ELF
// RUN: %clang_cc1 -triple x86_64-fuchsia -emit-llvm %s -o - | FileCheck %s --check-prefixes=NO-IFUNC,NO-IFUNC-ELF
int __attribute__((target("sse4.2"))) foo(int i, ...) { return 0; }
int __attribute__((target("arch=sandybridge"))) foo(int i, ...);
int __attribute__((target("arch=ivybridge"))) foo(int i, ...) {return 1;}
int __attribute__((target("default"))) foo(int i, ...) { return 2; }

int bar(void) {
  return foo(1, 'a', 1.1) + foo(2, 2.2, "asdf");
}

// IFUNC-ELF: @foo.ifunc = weak_odr ifunc i32 (i32, ...), ptr @foo.resolver
// IFUNC-ELF: define{{.*}} i32 @foo.sse4.2(i32 noundef %i, ...)
// IFUNC-ELF: ret i32 0
// IFUNC-ELF: define{{.*}} i32 @foo.arch_ivybridge(i32 noundef %i, ...)
// IFUNC-ELF: ret i32 1
// IFUNC-ELF: define{{.*}} i32 @foo(i32 noundef %i, ...)
// IFUNC-ELF: ret i32 2
// IFUNC-ELF: define{{.*}} i32 @bar()
// IFUNC-ELF: call i32 (i32, ...) @foo.ifunc(i32 noundef 1, i32 noundef 97, double
// IFUNC-ELF: call i32 (i32, ...) @foo.ifunc(i32 noundef 2, double noundef 2.2{{[0-9Ee+]+}}, ptr noundef

// IFUNC-ELF: define weak_odr ptr @foo.resolver() comdat
// IFUNC-ELF: ret ptr @foo.arch_sandybridge
// IFUNC-ELF: ret ptr @foo.arch_ivybridge
// IFUNC-ELF: ret ptr @foo.sse4.2
// IFUNC-ELF: ret ptr @foo
// IFUNC-ELF: declare i32 @foo.arch_sandybridge(i32 noundef, ...)

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
// NO-IFUNC-ELF: define weak_odr i32 @foo.resolver(i32 %0, ...) comdat
// NO-IFUNC: musttail call i32 (i32, ...) @foo.arch_sandybridge
// NO-IFUNC: musttail call i32 (i32, ...) @foo.arch_ivybridge
// NO-IFUNC: musttail call i32 (i32, ...) @foo.sse4.2
// NO-IFUNC: musttail call i32 (i32, ...) @foo
// WINDOWS: declare dso_local i32 @foo.arch_sandybridge(i32 noundef, ...)
// NO-IFUNC-ELF: declare i32 @foo.arch_sandybridge(i32 noundef, ...)
