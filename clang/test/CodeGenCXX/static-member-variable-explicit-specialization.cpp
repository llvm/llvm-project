// RUN: %clang_cc1 %s -std=c++1y -triple=x86_64-pc-linux -emit-llvm -o - | FileCheck --check-prefix=ELF --check-prefix=ALL %s
// RUN: %clang_cc1 %s -std=c++1y -triple=x86_64-apple-darwin -emit-llvm -o - | FileCheck --check-prefix=MACHO --check-prefix=ALL %s

// ALL: ; ModuleID

extern "C" int foo();

template<typename T> struct A { static int a; };
template<typename T> int A<T>::a = foo();

// ALLK-NOT: @_ZN1AIcE1aE
template<> int A<char>::a;

// ALL: @_ZN1AIbE1aE = global i32 10
template<> int A<bool>::a = 10;

// ALL: @llvm.global_ctors = appending global [8 x { i32, void ()*, i8* }]

// ELF: [{ i32, void ()*, i8* } { i32 65535, void ()* @[[unordered1:[^,]*]], i8* bitcast (i32* @_ZN1AIsE1aE to i8*) },
// MACHO: [{ i32, void ()*, i8* } { i32 65535, void ()* @[[unordered1:[^,]*]], i8* null },

// ELF:  { i32, void ()*, i8* } { i32 65535, void ()* @[[unordered2:[^,]*]], i8* bitcast (i16* @_Z1xIsE to i8*) },
// MACHO:  { i32, void ()*, i8* } { i32 65535, void ()* @[[unordered2:[^,]*]], i8* null },

// ELF:  { i32, void ()*, i8* } { i32 65535, void ()* @[[unordered3:[^,]*]], i8* bitcast (i32* @_ZN2ns1aIiE1iE to i8*) },
// MACHO:  { i32, void ()*, i8* } { i32 65535, void ()* @[[unordered3:[^,]*]], i8* null },

// ELF:  { i32, void ()*, i8* } { i32 65535, void ()* @[[unordered4:[^,]*]], i8* bitcast (i32* @_ZN2ns1b1iIiEE to i8*) },
// MACHO:  { i32, void ()*, i8* } { i32 65535, void ()* @[[unordered4:[^,]*]], i8* null },

// ELF:  { i32, void ()*, i8* } { i32 65535, void ()* @[[unordered5:[^,]*]], i8* bitcast (i32* @_ZN1AIvE1aE to i8*) },
// MACHO:  { i32, void ()*, i8* } { i32 65535, void ()* @[[unordered5:[^,]*]], i8* null },

// ELF:  { i32, void ()*, i8* } { i32 65535, void ()* @[[unordered6:[^,]*]], i8* @_Z1xIcE },
// MACHO:  { i32, void ()*, i8* } { i32 65535, void ()* @[[unordered6:[^,]*]], i8* null },

// ALL:  { i32, void ()*, i8* } { i32 65535, void ()* @[[unordered7:[^,]*]], i8* null },

// ALL:  { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_static_member_variable_explicit_specialization.cpp, i8* null }]

template int A<short>::a;  // Unordered
int b = foo();
int c = foo();
int d = A<void>::a; // Unordered

// An explicit specialization is ordered, and goes in __GLOBAL_sub_I_static_member_variable_explicit_specialization.cpp.
template<> struct A<int> { static int a; };
int A<int>::a = foo();

template<typename T> struct S { static T x; static T y; };
template<> int S<int>::x = foo();
template<> int S<int>::y = S<int>::x;

template<typename T> T x = foo();
template short x<short>;  // Unordered
template<> int x<int> = foo();
int e = x<char>; // Unordered

namespace ns {
template <typename T> struct a {
  static int i;
};
template<typename T> int a<T>::i = foo();
template struct a<int>;

struct b {
  template <typename T> static T i;
};
template<typename T> T b::i = foo();
template int b::i<int>;
}

namespace {
template<typename T> struct Internal { static int a; };
template<typename T> int Internal<T>::a = foo();
}
int *use_internal_a = &Internal<int>::a;

// ALL: define internal void @[[unordered1]](
// ALL: call i32 @foo()
// ALL: store {{.*}} @_ZN1AIsE1aE
// ALL: ret

// ALL: define internal void @[[unordered2]](
// ALL: call i32 @foo()
// ALL: store {{.*}} @_Z1xIsE
// ALL: ret

// ALL: define internal void @[[unordered3]](
// ALL: call i32 @foo()
// ALL: store {{.*}} @_ZN2ns1aIiE1iE
// ALL: ret

// ALL: define internal void @[[unordered4]](
// ALL: call i32 @foo()
// ALL: store {{.*}} @_ZN2ns1b1iIiEE
// ALL: ret

// ALL: define internal void @[[unordered5]](
// ALL: call i32 @foo()
// ALL: store {{.*}} @_ZN1AIvE1aE
// ALL: ret

// ALL: define internal void @[[unordered6]](
// ALL: call i32 @foo()
// ALL: store {{.*}} @_Z1xIcE
// ALL: ret

// ALL: define internal void @[[unordered7]](
// ALL: call i32 @foo()
// ALL: store {{.*}} @_ZN12_GLOBAL__N_18InternalIiE1aE
// ALL: ret

// ALL: define internal void @_GLOBAL__sub_I_static_member_variable_explicit_specialization.cpp()
//   We call unique stubs for every ordered dynamic initializer in the TU.
// ALL: call
// ALL: call
// ALL: call
// ALL: call
// ALL: call
// ALL: call
// ALL: call
// ALL: call
// ALL-NOT: call
// ALL: ret
