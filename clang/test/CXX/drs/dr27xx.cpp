// RUN: %clang_cc1 -std=c++98 -verify=expected %s
// RUN: %clang_cc1 -std=c++11 -verify=expected %s
// RUN: %clang_cc1 -std=c++14 -verify=expected %s
// RUN: %clang_cc1 -std=c++17 -verify=expected %s
// RUN: %clang_cc1 -std=c++20 -verify=expected %s
// RUN: %clang_cc1 -std=c++23 -verify=expected,since-cxx23 %s
// RUN: %clang_cc1 -std=c++2c -verify=expected,since-cxx23,since-cxx26 %s

#if __cplusplus <= 202002L
// expected-no-diagnostics
#endif

namespace dr2789 { // dr2789: 18
#if __cplusplus >= 202302L
template <typename T = int>
struct Base {
    constexpr void g(); // #dr2789-g1
};

template <typename T = int>
struct Base2 {
    constexpr void g() requires true;  // #dr2789-g2
};

template <typename T = int>
struct S : Base<T>, Base2<T> {
    constexpr void f();
    constexpr void f(this S&) requires true{};

    using Base<T>::g;
    using Base2<T>::g;
};

void test() {
    S<> s;
    s.f();
    s.g();
    // since-cxx23-error@-1 {{call to member function 'g' is ambiguous}}
    //   since-cxx23-note@#dr2789-g1 {{candidate function}}
    //   since-cxx23-note@#dr2789-g2 {{candidate function}}
}
#endif
}

namespace dr2798 { // dr2798: 17
#if __cpp_static_assert >= 202306
struct string {
  constexpr string() {
    data_ = new char[6]();
    __builtin_memcpy(data_, "Hello", 5);
    data_[5] = 0;
  }
  constexpr ~string() { delete[] data_; }
  constexpr unsigned long size() const { return 5; };
  constexpr const char *data() const { return data_; }

  char *data_;
};
struct X {
  string s;
};
consteval X f() { return {}; }

static_assert(false, f().s);
// since-cxx26-error@-1 {{static assertion failed: Hello}}
#endif
} // namespace dr2798

