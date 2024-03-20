// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++23 -emit-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -std=c++23 %t/b.cpp -fmodule-file=a=%t/a.pcm -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++23 -emit-reduced-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -std=c++23 %t/b.cpp -fmodule-file=a=%t/a.pcm -fsyntax-only -verify

//--- foo.h
class TypeA {};

template<class _Tp, class _Up>
concept __comparable = requires (_Tp &&__t, _Up &&__u) {
    __t == __u;
};

namespace ranges {
namespace __end {
  template <class _Tp>
  concept __member_end =
    requires(_Tp&& __t) {
        { __t.end() } -> __comparable<TypeA>;
    };

  struct __fn {
    template <class _Tp>
      requires __member_end<_Tp>
    constexpr auto operator()(_Tp&& __t) const
    {
      return true;
    }

    void operator()(auto&&) const = delete;
  };
}

inline namespace __cpo {
  inline constexpr auto end = __end::__fn{};
}
}

template <class _Tp>
concept range = requires(_Tp& __t) {
    ranges::end(__t);
};

template <class T>
class a {
public:
    a(T*) {}
    TypeA end() { return {}; }
};

template <class T>
class a_view {
public:
    template <class U>
    a_view(a<U>) {}
};
template <range _Range>
a_view(_Range) -> a_view<int>;

constexpr bool operator==(TypeA, TypeA) {
    return true;
}

//--- a.cppm
module;
#include "foo.h"
export module a;
export using ::a;
export using ::a_view;

//--- b.cpp
// expected-no-diagnostics
import a;
void use() {
    auto _ = a{"char"};
    auto __ = a_view{_};
}
