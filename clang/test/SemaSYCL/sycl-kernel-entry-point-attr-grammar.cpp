// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -fsyntax-only -fsycl-is-device -verify %s

// These tests validate parsing of the sycl_kernel_entry_point argument list
// and that the single argument names a type.

// Templates used to exercise class template specializations.
template<int> struct ST; // #ST-decl
template<int N> using TTA = ST<N>; // #TTA-decl


////////////////////////////////////////////////////////////////////////////////
// Valid declarations.
////////////////////////////////////////////////////////////////////////////////

struct S1;
[[clang::sycl_kernel_entry_point(S1)]] void ok1();

typedef struct {} TA2;
[[clang::sycl_kernel_entry_point(TA2)]] void ok2();

using TA3 = struct {};
[[clang::sycl_kernel_entry_point(TA3)]] void ok3();

[[clang::sycl_kernel_entry_point(ST<4>)]] void ok4();

[[clang::sycl_kernel_entry_point(TTA<5>)]] void ok5();

namespace NS6 {
  struct NSS;
}
[[clang::sycl_kernel_entry_point(NS6::NSS)]] void ok6();

namespace {
  struct UNSS7;
}
[[clang::sycl_kernel_entry_point(UNSS7)]] void ok7();

struct {} s;
[[clang::sycl_kernel_entry_point(decltype(s))]] void ok8();

template<typename KN>
[[clang::sycl_kernel_entry_point(KN)]] void ok9();
void test_ok9() {
  ok9<struct LS1>();
}

template<int, typename KN>
[[clang::sycl_kernel_entry_point(KN)]] void ok10();
void test_ok10() {
  ok10<1, struct LS2>();
}

namespace NS11 {
  struct NSS;
}
template<typename T>
[[clang::sycl_kernel_entry_point(T)]] void ok11() {}
template<>
[[clang::sycl_kernel_entry_point(NS11::NSS)]] void ok11<NS11::NSS>() {}

struct S12;
[[clang::sycl_kernel_entry_point(S12)]] void ok12();
[[clang::sycl_kernel_entry_point(S12)]] void ok12() {}

template<typename T>
[[clang::sycl_kernel_entry_point(T)]] void ok13(T k);
void test_ok13() {
  ok13([]{});
}


////////////////////////////////////////////////////////////////////////////////
// Invalid declarations.
////////////////////////////////////////////////////////////////////////////////

// expected-error@+1 {{'sycl_kernel_entry_point' attribute takes one argument}}
[[clang::sycl_kernel_entry_point]] void bad1();

// expected-error@+1 {{'sycl_kernel_entry_point' attribute takes one argument}}
[[clang::sycl_kernel_entry_point()]] void bad2();

struct B3;
// expected-error@+2 {{expected ')'}}
// expected-error@+1 {{expected ']'}}
[[clang::sycl_kernel_entry_point(B3,)]] void bad3();

struct B4;
// expected-error@+3 {{expected ')'}}
// expected-error@+2 {{expected ','}}
// expected-warning@+1 {{unknown attribute 'X' ignored}}
[[clang::sycl_kernel_entry_point(B4, X)]] void bad4();

// expected-error@+1 {{expected a type}}
[[clang::sycl_kernel_entry_point(1)]] void bad5();

void f6();
// expected-error@+1 {{unknown type name 'f6'}}
[[clang::sycl_kernel_entry_point(f6)]] void bad6();

// expected-error@+2 {{use of class template 'ST' requires template arguments; argument deduction not allowed here}}
// expected-note@#ST-decl {{template is declared here}}
[[clang::sycl_kernel_entry_point(ST)]] void bad7();

// expected-error@+2 {{use of alias template 'TTA' requires template arguments; argument deduction not allowed here}}
// expected-note@#TTA-decl {{template is declared here}}
[[clang::sycl_kernel_entry_point(TTA)]] void bad8();

enum {
  e9
};
// expected-error@+1 {{unknown type name 'e9'}}
[[clang::sycl_kernel_entry_point(e9)]] void bad9();

#if __cplusplus >= 202002L
template<typename> concept C = true;
// expected-error@+1 {{expected a type}}
[[clang::sycl_kernel_entry_point(C)]] void bad10();

// expected-error@+1 {{expected a type}}
[[clang::sycl_kernel_entry_point(C<int>)]] void bad11();
#endif
