// RUN: %clang_cc1 -fsyntax-only -verify %s

// libstdc++ 4.2.x contains a bug where a friend struct template
// declaration for std::tr1::__detail::_Map base has different
// template arguments than the real declaration.

// We no longer contain the hack to workaround the problem.  Verify that
// std::tr1::__detail::Map_base is not a unique and special snowflake.

namespace std { namespace tr1 { namespace __detail {
template <typename _Key, typename _Value, typename _Ex, bool __unique,
          // expected-note@-1{{previous template declaration}}
          typename _Hashtable>
struct _Map_base {};
} } } 

namespace std { namespace tr1 {
  template<typename T>
  struct X1 {
    template <typename _Key2, typename _Pair, typename _Hashtable>
    // expected-error@-1{{too few template parameters}}
    friend struct __detail::_Map_base;
  };

} }

std::tr1::X1<int> x1i; // expected-note{{in instantiation}}
