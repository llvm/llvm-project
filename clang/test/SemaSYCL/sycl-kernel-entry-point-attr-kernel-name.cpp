// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -fsyntax-only -fsycl-is-device -verify %s

// These tests validate that the kernel name type argument provided to the
// sycl_kernel_entry_point attribute meets the requirements of a SYCL kernel
// name as described in section 5.2, "Naming of kernels", of the SYCL 2020
// specification.

struct S1;
// expected-warning@+3 {{redundant 'sycl_kernel_entry_point' attribute}}
// expected-note@+1  {{previous attribute is here}}
[[clang::sycl_kernel_entry_point(S1),
  clang::sycl_kernel_entry_point(S1)]]
void ok1();

// expected-error@+1 {{'int' is not a valid SYCL kernel name type; a non-union class type is required}}
[[clang::sycl_kernel_entry_point(int)]] void bad2();

// expected-error@+1 {{'int ()' is not a valid SYCL kernel name type; a non-union class type is required}}
[[clang::sycl_kernel_entry_point(int())]] void bad3();

// expected-error@+1 {{'int (*)()' is not a valid SYCL kernel name type; a non-union class type is required}}
[[clang::sycl_kernel_entry_point(int(*)())]] void bad4();

// expected-error@+1 {{'int (&)()' is not a valid SYCL kernel name type; a non-union class type is required}}
[[clang::sycl_kernel_entry_point(int(&)())]] void bad5();

// expected-error@+1 {{'decltype(nullptr)' (aka 'std::nullptr_t') is not a valid SYCL kernel name type; a non-union class type is required}}
[[clang::sycl_kernel_entry_point(decltype(nullptr))]] void bad6();

union U7; // #U7-decl
// expected-error@+2 {{'U7' is not a valid SYCL kernel name type; a non-union class type is required}}
// expected-note@#U7-decl {{'U7' declared here}}
[[clang::sycl_kernel_entry_point(U7)]] void bad7();

enum E8 {}; // #E8-decl
// expected-error@+2 {{'E8' is not a valid SYCL kernel name type; a non-union class type is required}}
// expected-note@#E8-decl {{'E8' declared here}}
[[clang::sycl_kernel_entry_point(E8)]] void bad8();

enum E9 : int; // #E9-decl
// expected-error@+2 {{'E9' is not a valid SYCL kernel name type; a non-union class type is required}}
// expected-note@#E9-decl {{'E9' declared here}}
[[clang::sycl_kernel_entry_point(E9)]] void bad9();

struct B10 {
  struct MS;
};
// FIXME-expected-error@+1 {{'sycl_kernel_entry_point' attribute argument must be a forward declarable class type}}
[[clang::sycl_kernel_entry_point(B10::MS)]] void bad10();

struct B11 {
  struct MS;
};
// FIXME-expected-error@+3 {{'sycl_kernel_entry_point' attribute argument must be a forward declarable class type}}
template<typename T>
[[clang::sycl_kernel_entry_point(typename T::MS)]] void bad11() {}
template void bad11<B11>();

template<typename T>
[[clang::sycl_kernel_entry_point(T)]] void bad12();
void f12() {
  // FIXME-expected-error@+2 {{'sycl_kernel_entry_point' attribute argument must be a forward declarable class type}}
  struct LS;
  bad12<LS>();
}

struct B13_1;
struct B13_2;
// expected-error@+3 {{'sycl_kernel_entry_point' kernel name argument does not match prior declaration: 'B13_2' vs 'B13_1'}}
// expected-note@+1  {{'bad13' declared here}}
[[clang::sycl_kernel_entry_point(B13_1)]] void bad13();
[[clang::sycl_kernel_entry_point(B13_2)]] void bad13() {}

struct B14_1;
struct B14_2;
// expected-error@+3 {{'sycl_kernel_entry_point' kernel name argument does not match prior declaration: 'B14_2' vs 'B14_1'}}
// expected-note@+1  {{previous attribute is here}}
[[clang::sycl_kernel_entry_point(B14_1),
  clang::sycl_kernel_entry_point(B14_2)]]
void bad14();

struct B15;
// expected-error@+3 {{'sycl_kernel_entry_point' kernel name argument conflicts with a previous declaration}}
// expected-note@+1  {{previous declaration is here}}
[[clang::sycl_kernel_entry_point(B15)]] void bad15_1();
[[clang::sycl_kernel_entry_point(B15)]] void bad15_2();

struct B16_1;
struct B16_2;
// expected-error@+4 {{'sycl_kernel_entry_point' kernel name argument does not match prior declaration: 'B16_2' vs 'B16_1'}}
// expected-note@+1  {{'bad16' declared here}}
[[clang::sycl_kernel_entry_point(B16_1)]] void bad16();
void bad16(); // The attribute from the previous declaration is inherited.
[[clang::sycl_kernel_entry_point(B16_2)]] void bad16();

template<int>
struct B17 {
  // expected-error@+1 {{'int' is not a valid SYCL kernel name type; a non-union class type is required}}
  [[clang::sycl_kernel_entry_point(int)]]
  static void bad17();
};

template<int>
struct B18 {
  // expected-error@+1 {{'int' is not a valid SYCL kernel name type; a non-union class type is required}}
  [[clang::sycl_kernel_entry_point(int)]]
  friend void bad18() {}
};

template<typename KNT>
struct B19 {
  // expected-error@+1 {{'int' is not a valid SYCL kernel name type; a non-union class type is required}}
  [[clang::sycl_kernel_entry_point(KNT)]]
  friend void bad19() {}
};
// expected-note@+1 {{in instantiation of template class 'B19<int>' requested here}}
B19<int> b19;
