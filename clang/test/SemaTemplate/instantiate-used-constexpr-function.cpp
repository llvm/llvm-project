// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace GH73232  {

template <typename _CharT>
struct basic_string {
  constexpr void _M_construct();
  constexpr basic_string() {
    _M_construct();
  }
};

basic_string<char> a;

template <typename _CharT>
constexpr void basic_string<_CharT>::_M_construct(){}
constexpr basic_string<char> str{};

template <typename T>
constexpr void g(T);

constexpr int f() { g(0); return 0; }

template <typename T>
constexpr void g(T) {}

constexpr int z = f();

}
