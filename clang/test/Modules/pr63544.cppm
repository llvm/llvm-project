// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++23 %t/a.cppm -emit-module-interface -o %t/m-a.pcm
// RUN: %clang_cc1 -std=c++23 %t/b.cppm -emit-module-interface -o %t/m-b.pcm
// RUN: %clang_cc1 -std=c++23 %t/m.cppm -emit-module-interface -o %t/m.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++23 %t/pr63544.cpp -fprebuilt-module-path=%t -fsyntax-only -verify

// Test again with reduced BMI.
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++23 %t/a.cppm -emit-reduced-module-interface -o %t/m-a.pcm
// RUN: %clang_cc1 -std=c++23 %t/b.cppm -emit-reduced-module-interface -o %t/m-b.pcm
// RUN: %clang_cc1 -std=c++23 %t/m.cppm -emit-reduced-module-interface -o %t/m.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++23 %t/pr63544.cpp -fprebuilt-module-path=%t -fsyntax-only -verify


//--- foo.h

namespace std {
struct strong_ordering {
  int n;
  constexpr operator int() const { return n; }
  static const strong_ordering equal, greater, less;
};
constexpr strong_ordering strong_ordering::equal = {0};
constexpr strong_ordering strong_ordering::greater = {1};
constexpr strong_ordering strong_ordering::less = {-1};
} // namespace std

namespace std {
template <typename _Tp>
class optional {
private:
    using value_type = _Tp;
    value_type __val_;
    bool __engaged_;
public:
    constexpr bool has_value() const noexcept
    {
        return this->__engaged_;
    }

    constexpr const value_type& operator*() const& noexcept
    {
        return __val_;
    }

    optional(_Tp v) : __val_(v) {
        __engaged_ = true;
    }
};

template <class _Tp>
concept __is_derived_from_optional = requires(const _Tp& __t) { []<class __Up>(const optional<__Up>&) {}(__t); };

template <class _Tp, class _Up>
    requires(!__is_derived_from_optional<_Up>)
constexpr strong_ordering
operator<=>(const optional<_Tp>& __x, const _Up& __v) {
    return __x.has_value() ? *__x <=> __v : strong_ordering::less;
}
} // namespace std

//--- a.cppm
module;
#include "foo.h"
export module m:a;
export namespace std {
    using std::optional;
    using std::operator<=>;
}

//--- b.cppm
module;
#include "foo.h"
export module m:b;
export namespace std {
    using std::optional;
    using std::operator<=>;
}

//--- m.cppm
export module m;
export import :a;
export import :b;

//--- pr63544.cpp
// expected-no-diagnostics
import m;
int pr63544() {
    std::optional<int> a(43);
    int t{3};
    return a<=>t;
}
