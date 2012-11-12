// RUN: %clang_cc1 -fsyntax-only %s

// libstdc++ 4.2.x contains a bug where a friend struct template
// declaration for std::tr1::__detail::_Map base has different
// template arguments than the real declaration. Clang has an
// egregious hack to work around this problem, since we can't modify
// all of the world's libstdc++'s.

namespace std { namespace tr1 { namespace __detail {
  template<typename _Key, typename _Value, typename _Ex, bool __unique,
	   typename _Hashtable>
    struct _Map_base { };

} } } 

namespace std { namespace tr1 {
  template<typename T>
  struct X1 {
    template<typename _Key2, typename _Pair, typename _Hashtable>
    friend struct __detail::_Map_base;
  };

} }

std::tr1::X1<int> x1i;
