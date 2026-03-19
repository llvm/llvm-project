// RUN: %clang_cc1 -std=c++20 -Wno-ignored-attributes -Wno-unused-value -verify %s
// expected-no-diagnostics
namespace std {
template <class T>
constexpr const T& as_const(T&) noexcept;

// We need two declarations to see the error for some reason.
template <class T> void as_const(const T&&) noexcept = delete;
template <class T> void as_const(const T&&) noexcept;
}

namespace GH126231 {

void test() {
    int a = 1;
    std::as_const(a);
}
}
