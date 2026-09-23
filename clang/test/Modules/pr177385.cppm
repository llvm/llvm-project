// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
// RUN: mkdir %t/tmp

// RUN: %clang_cc1 -std=c++26 %t/std.cppm  -emit-module-interface -o %t/std.pcm
// RUN: %clang_cc1 -std=c++26 %t/optional.cppm  -emit-module-interface -o %t/optional.pcm \
// RUN:       -fmodule-file=std=%t/std.pcm
// RUN: %clang_cc1 -std=c++26 %t/date.format.cppm  -emit-module-interface -o %t/date.format.pcm \
// RUN:       -fmodule-file=std=%t/std.pcm \
// RUN:       -fmodule-file=bopt.optional=%t/optional.pcm
// RUN: %clang_cc1 -std=c++26 %t/annotated.cppm  -emit-module-interface -o %t/annotated.pcm \
// RUN:       -fmodule-file=std=%t/std.pcm \
// RUN:       -fmodule-file=bopt.optional=%t/optional.pcm \
// RUN:       -fmodule-file=ldgr:date.format=%t/date.format.pcm
// RUN: %clang_cc1 -std=c++26 %t/annotated.format.cppm  -emit-module-interface -o %t/annotated.format.pcm \
// RUN:       -fmodule-file=std=%t/std.pcm \
// RUN:       -fmodule-file=bopt.optional=%t/optional.pcm \
// RUN:       -fmodule-file=ldgr:date.format=%t/date.format.pcm \
// RUN:       -fmodule-file=ldgr:annotated=%t/annotated.pcm
// RUN: %clang_cc1 -std=c++26 %t/date.test.cppm  -emit-module-interface -o %t/date.test.pcm \
// RUN:       -fmodule-file=std=%t/std.pcm \
// RUN:       -fmodule-file=bopt.optional=%t/optional.pcm \
// RUN:       -fmodule-file=ldgr:date.format=%t/date.format.pcm \
// RUN:       -fmodule-file=ldgr:annotated=%t/annotated.pcm \
// RUN:       -fmodule-file=ldgr:annotated.format=%t/annotated.format.pcm
// RUN: %clang_cc1 -std=c++26 -w %t/annotated.test.cpp -fsyntax-only -verify \
// RUN:       -fmodule-file=std=%t/std.pcm \
// RUN:       -fmodule-file=bopt.optional=%t/optional.pcm \
// RUN:       -fmodule-file=ldgr:date.format=%t/date.format.pcm \
// RUN:       -fmodule-file=ldgr:annotated=%t/annotated.pcm \
// RUN:       -fmodule-file=ldgr:annotated.format=%t/annotated.format.pcm \
// RUN:       -fmodule-file=ldgr:date.testlib=%t/date.test.pcm

//--- std.cppm
module;
namespace std {
inline namespace __1 {
template <class _Tp> _Tp forward(_Tp);
template <class... _Args>
void __invoke(_Args... __args) noexcept(
    noexcept(__builtin_invoke(forward(__args)...)));
using string = char;
struct in_place_t {
} in_place;
template <typename...> struct __traits;
struct Trans_NS___visitation___base {
  template <class _Visitor, class... _Vs>
  static void __visit_alt(_Visitor, _Vs... __vs) {
    __make_fmatrix<_Visitor, decltype(__vs)...>;
  }
  struct __dispatcher {
    template <class _Fp, class... _Vs> static void __dispatch(_Vs... __vs) {
      _Fp __f;
      __invoke(__f, __vs...);
    }
  };
  template <class _Fp, class... _Vs> static void __make_dispatch() {
    __dispatcher::__dispatch<_Fp, _Vs...>;
  }
  template <class _Fp, class... _Vs> static void __make_fmatrix_impl() {
    __make_dispatch<_Fp, _Vs...>;
  }
  template <class _Fp, class... _Vs> static void __make_fmatrix() {
    __make_fmatrix_impl<_Fp, _Vs...>;
  }
};
template <class> class __dtor;
template <class... _Types> struct __dtor<__traits<_Types...>> {
  ~__dtor() {
    Trans_NS___visitation___base::__visit_alt([](auto) {}, this);
  }
};
struct __move_constructor : __dtor<__traits<>> {};
struct __assignment : __move_constructor {};
struct __impl : __assignment {};
template <class> struct variant {
  __impl __impl_;
};
}
}
export module std;
export namespace std {
using std::forward;
using std::in_place;
using std::in_place_t;
using std::string;
using std::variant;
}

//--- optional.cppm
export module bopt.optional;
import std;
namespace bopt {
template <class T> struct wrapper {
  template <class Func> wrapper(int, Func f) : value(std::forward(f)()) {}
  T value;
};
template <class T> struct optional_base {
  struct union_t {
    T value_;
  };
  struct repr {
    template <class OtherValue>
    repr(OtherValue) : un_{0, [] { return union_t{}; }} {}
    wrapper<union_t> un_;
  };
  template <class... Args>
  optional_base(std::in_place_t, Args... args) : repr_{args...} {}
  repr repr_;
};
export template <class T> struct optional : optional_base<T> {
  optional();
  template <class U> optional(U) : optional_base<T>{std::in_place, 0} {}
  template <class U> constexpr void operator=(U) {
    [] {};
  }
};
}

//--- date.format.cppm
module ldgr:date.format;
import bopt.optional;
namespace ldgr {
struct date_format {
  template <class S> static int parse(S s) { bopt::optional<int> year{s}; }
};
}

//--- annotated.cppm
module ldgr:annotated;
import std;
import bopt.optional;
struct lot_annotation {
  using valuation_expr_t = std::variant<std::string>;
  bopt::optional<valuation_expr_t> valuation_expr_;
};
struct annotated_amount {
  lot_annotation lot_annotation_;
};

//--- annotated.format.cppm
module ldgr:annotated.format;
import :date.format;
import :annotated;
namespace ldgr {
struct annotated_format {
  static annotated_amount parse_default() {
    bopt::optional<int> date;
    date = date_format::parse(0);
  }
};
}

//--- date.test.cppm
module ldgr:date.testlib;
import :date.format;
void date_from_string() { ldgr::date_format::parse(0); }

//--- annotated.test.cpp
// expected-no-diagnostics
module ldgr:annotatedtest;
import :annotated.format;
import :date.testlib;
int main() { ldgr::annotated_format::parse_default(); }
