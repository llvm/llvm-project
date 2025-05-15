// RUN: %clang_cc1 -std=c++03 %s "-DTYPE_CAST=" -verify                -Wno-unused -Wno-c++23-extensions -Wno-c++20-extensions -Wno-c++17-extensions -Wno-c++14-extensions -Wno-c++11-extensions
// RUN: %clang_cc1 -std=c++11 %s "-DTYPE_CAST=" -verify=expected,cxx11 -Wno-unused -Wno-c++23-extensions -Wno-c++20-extensions -Wno-c++17-extensions -Wno-c++14-extensions
// RUN: %clang_cc1 -std=c++14 %s "-DTYPE_CAST=" -verify                -Wno-unused -Wno-c++23-extensions -Wno-c++20-extensions -Wno-c++17-extensions
// RUN: %clang_cc1 -std=c++17 %s "-DTYPE_CAST=" -verify                -Wno-unused -Wno-c++23-extensions -Wno-c++20-extensions
// RUN: %clang_cc1 -std=c++20 %s "-DTYPE_CAST=" -verify                -Wno-unused -Wno-c++23-extensions
// RUN: %clang_cc1 -std=c++23 %s "-DTYPE_CAST=" -verify                -Wno-unused

// RUN: %clang_cc1 -std=c++03 %s "-DTYPE_CAST=(void)" -verify                -Wno-unused -Wno-c++23-extensions -Wno-c++20-extensions -Wno-c++17-extensions -Wno-c++14-extensions -Wno-c++11-extensions
// RUN: %clang_cc1 -std=c++11 %s "-DTYPE_CAST=(void)" -verify=expected,cxx11 -Wno-unused -Wno-c++23-extensions -Wno-c++20-extensions -Wno-c++17-extensions -Wno-c++14-extensions
// RUN: %clang_cc1 -std=c++14 %s "-DTYPE_CAST=(void)" -verify                -Wno-unused -Wno-c++23-extensions -Wno-c++20-extensions -Wno-c++17-extensions
// RUN: %clang_cc1 -std=c++17 %s "-DTYPE_CAST=(void)" -verify                -Wno-unused -Wno-c++23-extensions -Wno-c++20-extensions
// RUN: %clang_cc1 -std=c++20 %s "-DTYPE_CAST=(void)" -verify                -Wno-unused -Wno-c++23-extensions
// RUN: %clang_cc1 -std=c++23 %s "-DTYPE_CAST=(void)" -verify                -Wno-unused

void test() {

TYPE_CAST [] {};
TYPE_CAST []() {};
TYPE_CAST []() mutable {};
#if __cplusplus >= 201103L
TYPE_CAST []() constexpr {}; // cxx11-error {{return type 'void' is not a literal type}}
#endif

#if __cplusplus >= 201103L
TYPE_CAST [] constexpr {}; // cxx11-error {{return type 'void' is not a literal type}}
#endif
TYPE_CAST [] mutable {};
#if __cplusplus >= 201103L
TYPE_CAST [] noexcept {};
TYPE_CAST [] constexpr mutable {}; // cxx11-error {{return type 'void' is not a literal type}}
TYPE_CAST [] mutable constexpr {}; // cxx11-error {{return type 'void' is not a literal type}}
TYPE_CAST [] constexpr mutable noexcept {}; // cxx11-error {{return type 'void' is not a literal type}}
#endif
TYPE_CAST [s = 1] mutable {};
#if __cplusplus >= 201103L
TYPE_CAST [s = 1] constexpr mutable noexcept {}; // cxx11-error {{return type 'void' is not a literal type}}
#endif
TYPE_CAST [] -> bool { return true; };
TYPE_CAST []<typename T> { return true; };
#if __cplusplus >= 201103L
TYPE_CAST []<typename T> noexcept { return true; };
#endif
TYPE_CAST []<typename T> -> bool { return true; };
#if __cplusplus >= 202002L
TYPE_CAST [] consteval {};
TYPE_CAST []() requires true {}; // expected-error{{non-templated function cannot have a requires clause}}
TYPE_CAST []<auto> requires true() requires true {};
TYPE_CAST []<auto> requires true noexcept {};
#endif
TYPE_CAST [] [[maybe_unused]]{};

#if __cplusplus >= 201103L
TYPE_CAST [] mutable constexpr mutable {};    // expected-error{{cannot appear multiple times}} cxx11-error {{return type 'void' is not a literal type}}
TYPE_CAST [] constexpr mutable constexpr {};  // expected-error{{cannot appear multiple times}} cxx11-error {{return type 'void' is not a literal type}}

[]) constexpr mutable constexpr {}; // expected-error{{expected body of lambda expression}}
[]( constexpr mutable constexpr {}; // expected-error{{invalid storage class specifier}} \
                                    // expected-error{{function parameter cannot be constexpr}} \
                                    // expected-error{{a type specifier is required}} \
                                    // expected-error{{expected ')'}} \
                                    // expected-note{{to match this '('}} \
                                    // expected-error{{expected body}} \
                                    // expected-warning{{duplicate 'constexpr'}}

#endif

// http://llvm.org/PR49736

#if __cplusplus >= 202002L
[] requires true {}; // expected-error{{expected body}}
(void)[] requires true {}; // expected-error{{expected body}}
#else
[] requires true {}; // expected-error{{expected body}}
(void)[] requires true {}; // expected-error{{expected expression}}
#endif

#if __cplusplus >= 201703L
TYPE_CAST []<auto> requires true requires true {}; // expected-error{{expected body}}
TYPE_CAST []<auto> requires true noexcept requires true {}; // expected-error{{expected body}}
#endif

TYPE_CAST []() static static {}; // expected-error {{cannot appear multiple times}}
TYPE_CAST []() static mutable {}; // expected-error {{cannot be both mutable and static}}
#if __cplusplus >= 202002L
TYPE_CAST []() static consteval {};
#endif
#if __cplusplus >= 201103L
TYPE_CAST []() static constexpr {}; // cxx11-error {{return type 'void' is not a literal type}}
#endif

TYPE_CAST [] static {};
TYPE_CAST []() static {};
TYPE_CAST []() static extern {};  // expected-error {{expected body of lambda expression}}
TYPE_CAST []() extern {};  // expected-error {{expected body of lambda expression}}

}


void static_captures() {
  int x;
  TYPE_CAST [&]() static {}; // expected-error {{a static lambda cannot have any captures}}
  TYPE_CAST [x]() static {}; // expected-error {{a static lambda cannot have any captures}}
  TYPE_CAST [&x]() static {}; // expected-error {{a static lambda cannot have any captures}}
  TYPE_CAST [y=x]() static {}; // expected-error {{a static lambda cannot have any captures}}
  TYPE_CAST [&y = x]() static {}; // expected-error {{a static lambda cannot have any captures}}
  TYPE_CAST [=]() static {}; // expected-error {{a static lambda cannot have any captures}}
  struct X {
    int z;
    void f() {
      [this]() static {}(); // expected-error {{a static lambda cannot have any captures}}
      [*this]() static {}(); // expected-error {{a static lambda cannot have any captures}}
    }
  };
}

#if __cplusplus >= 201703L
constexpr auto static_capture_constexpr() {
  char n = 'n';
  return [n] static { return n; }(); // expected-error {{a static lambda cannot have any captures}}
}
static_assert(static_capture_constexpr()); // expected-error {{static assertion expression is not an integral constant expression}}

constexpr auto capture_constexpr() {
  char n = 'n';
  return [n] { return n; }();
}
static_assert(capture_constexpr());
#endif
