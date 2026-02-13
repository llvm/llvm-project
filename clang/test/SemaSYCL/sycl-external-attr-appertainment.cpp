// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -std=c++17 -verify %s
// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -std=c++17 -verify %s
// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -std=c++20 -verify %s
// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -std=c++20 -verify %s
// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -std=c++23 -verify %s
// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -std=c++23 -verify %s

// expected-error@+1{{'clang::sycl_external' attribute only applies to functions}}
[[clang::sycl_external]] int bad1;


// expected-error@+2{{'clang::sycl_external' attribute only applies to functions}}
struct s {
[[clang::sycl_external]] int bad2;
};

// expected-error@+1{{'clang::sycl_external' attribute only applies to functions}}
namespace [[clang::sycl_external]] bad3 {}

// expected-error@+1{{'clang::sycl_external' attribute only applies to functions}}
struct [[clang::sycl_external]]  bad4;

// expected-error@+1{{'clang::sycl_external' attribute only applies to functions}}
enum [[clang::sycl_external]] bad5 {};

// expected-error@+1{{'clang::sycl_external' attribute only applies to functions}}
int bad6(void (fp [[clang::sycl_external]])());

// expected-error@+1{{'clang::sycl_external' attribute only applies to functions}}
[[clang::sycl_external]];

#if __cplusplus >= 202002L
// expected-error@+2{{'clang::sycl_external' attribute only applies to functions}}
template<typename>
concept bad8 [[clang::sycl_external]] = true;
#endif
