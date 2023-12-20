// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=ITANIUM,LINUX
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-macos -emit-llvm %s -o - | FileCheck %s --check-prefixes=ITANIUM,DARWIN
// RUN: %clang_cc1 -std=c++11 -triple x86_64-windows-pc -emit-llvm %s -o - | FileCheck %s --check-prefix=WINDOWS

// DARWIN-NOT: comdat

// Aliases for ifuncs
// ITANIUM: @_Z10overloadedi.ifunc = weak_odr alias i32 (i32), ptr @_Z10overloadedi
// ITANIUM: @_Z10overloadedPKc.ifunc = weak_odr alias i32 (ptr), ptr @_Z10overloadedPKc
// ITANIUM: @_ZN1CIssE3fooEv.ifunc = weak_odr alias i32 (ptr), ptr @_ZN1CIssE3fooEv
// ITANIUM: @_ZN1CIisE3fooEv.ifunc = weak_odr alias i32 (ptr), ptr @_ZN1CIisE3fooEv
// ITANIUM: @_ZN1CIdfE3fooEv.ifunc = weak_odr alias i32 (ptr), ptr @_ZN1CIdfE3fooEv

// Overloaded ifuncs
// ITANIUM: @_Z10overloadedi = weak_odr ifunc i32 (i32), ptr @_Z10overloadedi.resolver
// ITANIUM: @_Z10overloadedPKc = weak_odr ifunc i32 (ptr), ptr @_Z10overloadedPKc.resolver
// struct 'C' ifuncs, note the 'float, U' one doesn't get one.
// ITANIUM: @_ZN1CIssE3fooEv = weak_odr ifunc i32 (ptr), ptr @_ZN1CIssE3fooEv.resolver
// ITANIUM: @_ZN1CIisE3fooEv = weak_odr ifunc i32 (ptr), ptr @_ZN1CIisE3fooEv.resolver
// ITANIUM: @_ZN1CIdfE3fooEv = weak_odr ifunc i32 (ptr), ptr @_ZN1CIdfE3fooEv.resolver

int __attribute__((target_clones("sse4.2", "default"))) overloaded(int) { return 1; }
// ITANIUM: define {{.*}}i32 @_Z10overloadedi.sse4.2.0(i32{{.+}})
// ITANIUM: define {{.*}}i32 @_Z10overloadedi.default.1(i32{{.+}})
// ITANIUM: define weak_odr ptr @_Z10overloadedi.resolver()
// LINUX-SAME: comdat
// ITANIUM: ret ptr @_Z10overloadedi.sse4.2.0
// ITANIUM: ret ptr @_Z10overloadedi.default.1

// WINDOWS: define dso_local noundef i32 @"?overloaded@@YAHH@Z.sse4.2.0"(i32{{.+}})
// WINDOWS: define dso_local noundef i32 @"?overloaded@@YAHH@Z.default.1"(i32{{.+}})
// WINDOWS: define weak_odr dso_local i32 @"?overloaded@@YAHH@Z"(i32{{.+}}) comdat
// WINDOWS: call i32 @"?overloaded@@YAHH@Z.sse4.2.0"
// WINDOWS: call i32 @"?overloaded@@YAHH@Z.default.1"

int __attribute__((target_clones("arch=ivybridge", "default"))) overloaded(const char *) { return 2; }
// ITANIUM: define {{.*}}i32 @_Z10overloadedPKc.arch_ivybridge.0(ptr{{.+}})
// ITANIUM: define {{.*}}i32 @_Z10overloadedPKc.default.1(ptr{{.+}})
// ITANIUM: define weak_odr ptr @_Z10overloadedPKc.resolver()
// LINUX-SAME: comdat
// ITANIUM: ret ptr @_Z10overloadedPKc.arch_ivybridge.0
// ITANIUM: ret ptr @_Z10overloadedPKc.default.1

// WINDOWS: define dso_local noundef i32 @"?overloaded@@YAHPEBD@Z.arch_ivybridge.0"(ptr{{.+}})
// WINDOWS: define dso_local noundef i32 @"?overloaded@@YAHPEBD@Z.default.1"(ptr{{.+}})
// WINDOWS: define weak_odr dso_local i32 @"?overloaded@@YAHPEBD@Z"(ptr{{.+}}) comdat
// WINDOWS: call i32 @"?overloaded@@YAHPEBD@Z.arch_ivybridge.0"
// WINDOWS: call i32 @"?overloaded@@YAHPEBD@Z.default.1"

void use_overloaded() {
  overloaded(1);
  // ITANIUM: call noundef i32 @_Z10overloadedi
  // WINDOWS: call noundef i32 @"?overloaded@@YAHH@Z"
  overloaded(nullptr);
  // ITANIUM: call noundef i32 @_Z10overloadedPKc 
  // WINDOWS: call noundef i32 @"?overloaded@@YAHPEBD@Z"
}

template<typename T, typename U>
struct C {
int __attribute__((target_clones("sse4.2", "default"))) foo(){ return 1;}
};
template<typename U>
struct C<int, U> {
int __attribute__((target_clones("sse4.2", "default"))) foo(){ return 2;}
};
template<typename U>
struct C<float, U> {
int foo(){ return 2;}
};
template<>
struct C<double, float> {
int __attribute__((target_clones("sse4.2", "default"))) foo(){ return 3;}
};

void uses_specialized() {
  C<short, short> c;
  c.foo();
  // ITANIUM: call noundef i32 @_ZN1CIssE3fooEv(ptr
  // WINDOWS: call noundef i32 @"?foo@?$C@FF@@QEAAHXZ"(ptr
  C<int, short> c2;
  c2.foo();
  // ITANIUM: call noundef i32 @_ZN1CIisE3fooEv(ptr
  // WINDOWS: call noundef i32 @"?foo@?$C@HF@@QEAAHXZ"(ptr
  C<float, short> c3;
  c3.foo();
  // Note this is not an ifunc/mv
  // ITANIUM: call noundef i32 @_ZN1CIfsE3fooEv(ptr
  // WINDOWS: call noundef i32 @"?foo@?$C@MF@@QEAAHXZ"(ptr
  C<double, float> c4;
  c4.foo();
  // ITANIUM: call noundef i32 @_ZN1CIdfE3fooEv(ptr
  // WINDOWS: call noundef i32 @"?foo@?$C@NM@@QEAAHXZ"(ptr
}

// ITANIUM: define weak_odr ptr @_ZN1CIssE3fooEv.resolver()
// LINUX-SAME: comdat
// ITANIUM: ret ptr @_ZN1CIssE3fooEv.sse4.2.0
// ITANIUM: ret ptr @_ZN1CIssE3fooEv.default.1

// WINDOWS: define {{.*}}i32 @"?foo@?$C@FF@@QEAAHXZ"(ptr
// WINDOWS: call i32 @"?foo@?$C@FF@@QEAAHXZ.sse4.2.0"
// WINDOWS: call i32 @"?foo@?$C@FF@@QEAAHXZ.default.1"

// ITANIUM: define weak_odr ptr @_ZN1CIisE3fooEv.resolver()
// LINUX-SAME: comdat
// ITANIUM: ret ptr @_ZN1CIisE3fooEv.sse4.2.0
// ITANIUM: ret ptr @_ZN1CIisE3fooEv.default.1

// WINDOWS: define {{.*}}i32 @"?foo@?$C@HF@@QEAAHXZ"(ptr
// WINDOWS: call i32 @"?foo@?$C@HF@@QEAAHXZ.sse4.2.0"
// WINDOWS: call i32 @"?foo@?$C@HF@@QEAAHXZ.default.1"

// ITANIUM: define weak_odr ptr @_ZN1CIdfE3fooEv.resolver()
// LINUX-SAME: comdat
// ITANIUM: ret ptr @_ZN1CIdfE3fooEv.sse4.2.0
// ITANIUM: ret ptr @_ZN1CIdfE3fooEv.default.1

// WINDOWS: define {{.*}}i32 @"?foo@?$C@NM@@QEAAHXZ"(ptr
// WINDOWS: call i32 @"?foo@?$C@NM@@QEAAHXZ.sse4.2.0"
// WINDOWS: call i32 @"?foo@?$C@NM@@QEAAHXZ.default.1"

// ITANIUM: define {{.*}}i32 @_ZN1CIssE3fooEv.sse4.2.0(ptr
// ITANIUM: define {{.*}}i32 @_ZN1CIssE3fooEv.default.1(ptr
// ITANIUM: define {{.*}}i32 @_ZN1CIisE3fooEv.sse4.2.0(ptr
// ITANIUM: define {{.*}}i32 @_ZN1CIisE3fooEv.default.1(ptr
// ITANIUM: define {{.*}}i32 @_ZN1CIdfE3fooEv.sse4.2.0(ptr
// ITANIUM: define {{.*}}i32 @_ZN1CIdfE3fooEv.default.1(ptr

// WINDOWS: define {{.*}}i32 @"?foo@?$C@FF@@QEAAHXZ.sse4.2.0"(ptr
// WINDOWS: define {{.*}}i32 @"?foo@?$C@FF@@QEAAHXZ.default.1"(ptr
// WINDOWS: define {{.*}}i32 @"?foo@?$C@HF@@QEAAHXZ.sse4.2.0"(ptr
// WINDOWS: define {{.*}}i32 @"?foo@?$C@HF@@QEAAHXZ.default.1"(ptr
// WINDOWS: define {{.*}}i32 @"?foo@?$C@NM@@QEAAHXZ.sse4.2.0"(ptr
// WINDOWS: define {{.*}}i32 @"?foo@?$C@NM@@QEAAHXZ.default.1"(ptr
