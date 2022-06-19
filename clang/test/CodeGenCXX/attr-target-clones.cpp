// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LINUX
// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -triple x86_64-windows-pc -emit-llvm %s -o - | FileCheck %s --check-prefix=WINDOWS

// Overloaded ifuncs
// LINUX: @_Z10overloadedi.ifunc = weak_odr ifunc i32 (i32), i32 (i32)* ()* @_Z10overloadedi.resolver
// LINUX: @_Z10overloadedPKc.ifunc = weak_odr ifunc i32 (i8*), i32 (i8*)* ()* @_Z10overloadedPKc.resolver
// struct 'C' ifuncs, note the 'float, U' one doesn't get one.
// LINUX: @_ZN1CIssE3fooEv.ifunc = weak_odr ifunc i32 (%struct.C*), i32 (%struct.C*)* ()* @_ZN1CIssE3fooEv.resolver
// LINUX: @_ZN1CIisE3fooEv.ifunc = weak_odr ifunc i32 (%struct.C.0*), i32 (%struct.C.0*)* ()* @_ZN1CIisE3fooEv.resolver
// LINUX: @_ZN1CIdfE3fooEv.ifunc = weak_odr ifunc i32 (%struct.C.2*), i32 (%struct.C.2*)* ()* @_ZN1CIdfE3fooEv.resolver

int __attribute__((target_clones("sse4.2", "default"))) overloaded(int) { return 1; }
// LINUX: define {{.*}}i32 @_Z10overloadedi.sse4.2.0(i32{{.+}})
// LINUX: define {{.*}}i32 @_Z10overloadedi.default.1(i32{{.+}})
// LINUX: define weak_odr i32 (i32)* @_Z10overloadedi.resolver() comdat
// LINUX: ret i32 (i32)* @_Z10overloadedi.sse4.2.0
// LINUX: ret i32 (i32)* @_Z10overloadedi.default.1

// WINDOWS: define dso_local noundef i32 @"?overloaded@@YAHH@Z.sse4.2.0"(i32{{.+}})
// WINDOWS: define dso_local noundef i32 @"?overloaded@@YAHH@Z.default.1"(i32{{.+}})
// WINDOWS: define weak_odr dso_local i32 @"?overloaded@@YAHH@Z"(i32{{.+}}) comdat
// WINDOWS: call i32 @"?overloaded@@YAHH@Z.sse4.2.0"
// WINDOWS: call i32 @"?overloaded@@YAHH@Z.default.1"

int __attribute__((target_clones("arch=ivybridge", "default"))) overloaded(const char *) { return 2; }
// LINUX: define {{.*}}i32 @_Z10overloadedPKc.arch_ivybridge.0(i8*{{.+}})
// LINUX: define {{.*}}i32 @_Z10overloadedPKc.default.1(i8*{{.+}})
// LINUX: define weak_odr i32 (i8*)* @_Z10overloadedPKc.resolver() comdat
// LINUX: ret i32 (i8*)* @_Z10overloadedPKc.arch_ivybridge.0
// LINUX: ret i32 (i8*)* @_Z10overloadedPKc.default.1

// WINDOWS: define dso_local noundef i32 @"?overloaded@@YAHPEBD@Z.arch_ivybridge.0"(i8*{{.+}})
// WINDOWS: define dso_local noundef i32 @"?overloaded@@YAHPEBD@Z.default.1"(i8*{{.+}})
// WINDOWS: define weak_odr dso_local i32 @"?overloaded@@YAHPEBD@Z"(i8*{{.+}}) comdat
// WINDOWS: call i32 @"?overloaded@@YAHPEBD@Z.arch_ivybridge.0"
// WINDOWS: call i32 @"?overloaded@@YAHPEBD@Z.default.1"

void use_overloaded() {
  overloaded(1);
  // LINUX: call noundef i32 @_Z10overloadedi.ifunc
  // WINDOWS: call noundef i32 @"?overloaded@@YAHH@Z"
  overloaded(nullptr);
  // LINUX: call noundef i32 @_Z10overloadedPKc.ifunc 
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
  // LINUX: call noundef i32 @_ZN1CIssE3fooEv.ifunc(%struct.C
  // WINDOWS: call noundef i32 @"?foo@?$C@FF@@QEAAHXZ"(%struct.C
  C<int, short> c2;
  c2.foo();
  // LINUX: call noundef i32 @_ZN1CIisE3fooEv.ifunc(%struct.C
  // WINDOWS: call noundef i32 @"?foo@?$C@HF@@QEAAHXZ"(%struct.C
  C<float, short> c3;
  c3.foo();
  // Note this is not an ifunc/mv
  // LINUX: call noundef i32 @_ZN1CIfsE3fooEv(%struct.C
  // WINDOWS: call noundef i32 @"?foo@?$C@MF@@QEAAHXZ"(%struct.C
  C<double, float> c4;
  c4.foo();
  // LINUX: call noundef i32 @_ZN1CIdfE3fooEv.ifunc(%struct.C
  // WINDOWS: call noundef i32 @"?foo@?$C@NM@@QEAAHXZ"(%struct.C
}

// LINUX: define weak_odr i32 (%struct.C*)* @_ZN1CIssE3fooEv.resolver() comdat
// LINUX: ret i32 (%struct.C*)* @_ZN1CIssE3fooEv.sse4.2.0
// LINUX: ret i32 (%struct.C*)* @_ZN1CIssE3fooEv.default.1

// WINDOWS: define {{.*}}i32 @"?foo@?$C@FF@@QEAAHXZ"(%struct.C{{.+}})
// WINDOWS: call i32 @"?foo@?$C@FF@@QEAAHXZ.sse4.2.0"
// WINDOWS: call i32 @"?foo@?$C@FF@@QEAAHXZ.default.1"

// LINUX: define weak_odr i32 (%struct.C{{.+}})* @_ZN1CIisE3fooEv.resolver() comdat
// LINUX: ret i32 (%struct.C{{.+}})* @_ZN1CIisE3fooEv.sse4.2.0
// LINUX: ret i32 (%struct.C{{.+}})* @_ZN1CIisE3fooEv.default.1

// WINDOWS: define {{.*}}i32 @"?foo@?$C@HF@@QEAAHXZ"(%struct.C{{.+}})
// WINDOWS: call i32 @"?foo@?$C@HF@@QEAAHXZ.sse4.2.0"
// WINDOWS: call i32 @"?foo@?$C@HF@@QEAAHXZ.default.1"

// LINUX: define weak_odr i32 (%struct.C{{.+}})* @_ZN1CIdfE3fooEv.resolver() comdat
// LINUX: ret i32 (%struct.C{{.+}})* @_ZN1CIdfE3fooEv.sse4.2.0
// LINUX: ret i32 (%struct.C{{.+}})* @_ZN1CIdfE3fooEv.default.1

// WINDOWS: define {{.*}}i32 @"?foo@?$C@NM@@QEAAHXZ"(%struct.C{{.+}})
// WINDOWS: call i32 @"?foo@?$C@NM@@QEAAHXZ.sse4.2.0"
// WINDOWS: call i32 @"?foo@?$C@NM@@QEAAHXZ.default.1"

// LINUX: define {{.*}}i32 @_ZN1CIssE3fooEv.sse4.2.0(%struct.C{{.+}})
// LINUX: define {{.*}}i32 @_ZN1CIssE3fooEv.default.1(%struct.C{{.+}})
// LINUX: define {{.*}}i32 @_ZN1CIisE3fooEv.sse4.2.0(%struct.C{{.+}})
// LINUX: define {{.*}}i32 @_ZN1CIisE3fooEv.default.1(%struct.C{{.+}})
// LINUX: define {{.*}}i32 @_ZN1CIdfE3fooEv.sse4.2.0(%struct.C{{.+}})
// LINUX: define {{.*}}i32 @_ZN1CIdfE3fooEv.default.1(%struct.C{{.+}})

// WINDOWS: define {{.*}}i32 @"?foo@?$C@FF@@QEAAHXZ.sse4.2.0"(%struct.C{{.+}})
// WINDOWS: define {{.*}}i32 @"?foo@?$C@FF@@QEAAHXZ.default.1"(%struct.C{{.+}})
// WINDOWS: define {{.*}}i32 @"?foo@?$C@HF@@QEAAHXZ.sse4.2.0"(%struct.C{{.+}})
// WINDOWS: define {{.*}}i32 @"?foo@?$C@HF@@QEAAHXZ.default.1"(%struct.C{{.+}})
// WINDOWS: define {{.*}}i32 @"?foo@?$C@NM@@QEAAHXZ.sse4.2.0"(%struct.C{{.+}})
// WINDOWS: define {{.*}}i32 @"?foo@?$C@NM@@QEAAHXZ.default.1"(%struct.C{{.+}})
