// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fclang-abi-compat=15 -DCLANG_ABI_COMPAT=15 %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,AFTER-15

// CHECK: %struct.S = type { i8 }
// CHECK: @_Z2ggiRi
// CHECK: @_Z1gIiJEERiPT_DpT0_
template <typename T, typename... U> int &g(T *, U...);
template <typename T> void g(T);
template <typename T, typename... Ts> struct S;
template <typename T> struct S<T> {};
void gg(int i, int &r) {
  r = g(&i);
  S<int> a;
}

// CHECK: @_Z1hIJiEEvDpPT_
template<class ...T> void h(T*...) {}
template<class T>    void h(const T&) {}
template void h(int*);

#if !defined(CLANG_ABI_COMPAT)

// AFTER-15: @_Z1fIiJEEvPT_DpT0_
template<class T, class... U> void f(T*, U...){}
template<class T> void f(T){}
template void f(int*);

template<class T, class... U> struct A;
template<class T1, class T2, class... U> struct A<T1,T2*,U...> {};
template<class T1, class T2> struct A<T1,T2>;
template struct A<int, int*>;

#endif
