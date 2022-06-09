// RUN: %clang_cc1 %s -std=c++1y -triple=x86_64-pc-linux -emit-llvm -o - | FileCheck --check-prefix=ELF --check-prefix=ALL %s
// RUN: %clang_cc1 %s -std=c++1y -triple=x86_64-apple-darwin -emit-llvm -o - | FileCheck --check-prefix=MACHO --check-prefix=ALL %s
// RUN: %clang_cc1 %s -std=c++1y -triple=x86_64-pc-linux -emit-llvm -fdeclspec -DSELECTANY -o - | FileCheck --check-prefix=ELF-SELECTANY %s

#ifdef SELECTANY
struct S {
  S();
  ~S();
};

int f();

// ELF-SELECTANY: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init, ptr @selectany }]
// ELF-SELECTANY: @llvm.used = appending global [1 x ptr] [ptr @selectany]
int __declspec(selectany) selectany = f();

#else

// ALL: ; ModuleID

extern "C" int foo();

template<typename T> struct A { static int a; };
template<typename T> int A<T>::a = foo();

// ALLK-NOT: @_ZN1AIcE1aE
template<> int A<char>::a;

// ALL: @_ZN1AIbE1aE ={{.*}} global i32 10
template<> int A<bool>::a = 10;

// ALL: @llvm.global_ctors = appending global [16 x { i32, ptr, ptr }]

// ELF:  [{ i32, ptr, ptr } { i32 65535, ptr @[[unordered:[^,]*]], ptr @_ZN1AIsE1aE },
// MACHO: [{ i32, ptr, ptr } { i32 65535, ptr @[[unordered:[^,]*]], ptr null },

// ELF:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered7:[^,]*]], ptr @_Z1xIsE },
// MACHO:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered7:[^,]*]], ptr null },

// ELF:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered10:[^,]*]], ptr @_ZN2ns1aIiE1iE },
// MACHO:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered10:[^,]*]], ptr null },

// ELF:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered11:[^,]*]], ptr @_ZN2ns1b1iIiEE },
// MACHO:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered11:[^,]*]], ptr null },

// ELF:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered14:[^,]*]], ptr @_ZN1AIvE1aE },
// MACHO:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered14:[^,]*]], ptr null },

// ELF:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered15:[^,]*]], ptr @_Z1xIcE },
// MACHO:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered15:[^,]*]], ptr null },

// ALL:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered16:[^,]*]], ptr null },

// ELF:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered19:[^,]*]], ptr @_ZN3FibILi2EE1aE },
// MACHO: { i32, ptr, ptr } { i32 65535, ptr @[[unordered19:[^,]*]], ptr null },

// ELF:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered18:[^,]*]], ptr @_ZN3FibILi3EE1aE },
// MACHO: { i32, ptr, ptr } { i32 65535, ptr @[[unordered18:[^,]*]], ptr null },

// ELF:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered20:[^,]*]], ptr @_ZN3FibILi4EE1aE },
// MACHO: { i32, ptr, ptr } { i32 65535, ptr @[[unordered20:[^,]*]], ptr null },

// ELF:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered17:[^,]*]], ptr @_ZN3FibILi5EE1aE },
// MACHO: { i32, ptr, ptr } { i32 65535, ptr @[[unordered17:[^,]*]], ptr null },

// ELF:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered23:[^,]*]], ptr @_ZN4Fib2ILi2EE1aE },
// MACHO: { i32, ptr, ptr } { i32 65535, ptr @[[unordered23:[^,]*]], ptr null },

// ELF:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered24:[^,]*]], ptr @_ZN4Fib2ILi3EE1aE },
// MACHO: { i32, ptr, ptr } { i32 65535, ptr @[[unordered24:[^,]*]], ptr null },

// ELF:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered22:[^,]*]], ptr @_ZN4Fib2ILi4EE1aE },
// MACHO: { i32, ptr, ptr } { i32 65535, ptr @[[unordered22:[^,]*]], ptr null },

// ELF:  { i32, ptr, ptr } { i32 65535, ptr @[[unordered21:[^,]*]], ptr @_ZN4Fib2ILi5EE1aE },
// MACHO: { i32, ptr, ptr } { i32 65535, ptr @[[unordered21:[^,]*]], ptr null }, 

// ALL:  { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_static_member_variable_explicit_specialization.cpp, ptr null }]

/// llvm.used ensures SHT_INIT_ARRAY in a section group cannot be GCed.
// ELF: @llvm.used = appending global [14 x ptr] [ptr @_ZN1AIsE1aE, ptr @_Z1xIsE, ptr @_ZN2ns1aIiE1iE, ptr @_ZN2ns1b1iIiEE, ptr @_ZN1AIvE1aE, ptr @_Z1xIcE, ptr @_ZN3FibILi5EE1aE, ptr @_ZN3FibILi3EE1aE, ptr @_ZN3FibILi2EE1aE, ptr @_ZN3FibILi4EE1aE, ptr @_ZN4Fib2ILi5EE1aE, ptr @_ZN4Fib2ILi4EE1aE, ptr @_ZN4Fib2ILi2EE1aE, ptr @_ZN4Fib2ILi3EE1aE]

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

template<int n> struct Fib { static int a; };
template<> int Fib<0>::a = 0;
template<> int Fib<1>::a = 1;
template<int n> int Fib<n>::a = Fib<n-2>::a + Fib<n-1>::a;
int f = Fib<5>::a;

template<int n> struct Fib2 { static int a; };
template<> int Fib2<0>::a = 0;
template<> int Fib2<1>::a = 1;
template<int n> int Fib2<n>::a = Fib2<n-1>::a + Fib2<n-2>::a;
int f2 = Fib2<5>::a;

#endif

// ALL: define internal void @[[unordered]](
// ALL: call i32 @foo()
// ALL: store {{.*}} @_ZN1AIsE1aE
// ALL: ret

// ALL: define internal void @[[unordered7]](
// ALL: call i32 @foo()
// ALL: store {{.*}} @_Z1xIsE
// ALL: ret

// ALL: define internal void @[[unordered10]](
// ALL: call i32 @foo()
// ALL: store {{.*}} @_ZN2ns1aIiE1iE
// ALL: ret

// ALL: define internal void @[[unordered11]](
// ALL: call i32 @foo()
// ALL: store {{.*}} @_ZN2ns1b1iIiEE
// ALL: ret

// ALL: define internal void @[[unordered15]](
// ALL: call i32 @foo()
// ALL: store {{.*}} @_Z1xIcE
// ALL: ret

// ALL: define internal void @[[unordered16]](
// ALL: call i32 @foo()
// ALL: store {{.*}} @_ZN12_GLOBAL__N_18InternalIiE1aE
// ALL: ret

// ALL: define internal void @[[unordered17]](
// ALL: store {{.*}} @_ZN3FibILi5EE1aE
// ALL: ret

// ALL: define internal void @[[unordered18]](
// ALL: store {{.*}} @_ZN3FibILi3EE1aE
// ALL: ret

// ALL: define internal void @[[unordered19]](
// ALL: store {{.*}} @_ZN3FibILi2EE1aE
// ALL: ret

// ALL: define internal void @[[unordered20]](
// ALL: store {{.*}} @_ZN3FibILi4EE1aE
// ALL: ret

// ALL: define internal void @[[unordered21]](
// ALL: store {{.*}} @_ZN4Fib2ILi5EE1aE
// ALL: ret

// ALL: define internal void @[[unordered22]](
// ALL: store {{.*}} @_ZN4Fib2ILi4EE1aE
// ALL: ret

// ALL: define internal void @[[unordered23]](
// ALL: store {{.*}} @_ZN4Fib2ILi2EE1aE
// ALL: ret

// ALL: define internal void @[[unordered24]](
// ALL: store {{.*}} @_ZN4Fib2ILi3EE1aE
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
// ALL: call
// ALL: call
// ALL-NOT: call
// ALL: ret
