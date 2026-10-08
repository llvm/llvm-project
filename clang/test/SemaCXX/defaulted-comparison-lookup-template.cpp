// RUN: %clang_cc1 -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++23 -verify %s
// RUN: %clang_cc1 -std=c++2c -verify %s

#include "Inputs/std-compare.h"

namespace Foreign {
struct S {
  int value;
};
struct Uncomparable {};
} // namespace Foreign

namespace Definition {
template <class T> struct C {
  T m;
  constexpr bool operator==(const C &) const;
};
// This must be found at the defaulted definition, not the class declaration.
constexpr bool operator==(Foreign::S a, Foreign::S b) {
  return a.value == b.value;
}
template <class T>
constexpr bool C<T>::operator==(const C &) const = default;
} // namespace Definition

namespace Caller {
bool operator==(Foreign::S, Foreign::S) = delete;
static_assert(Definition::C<Foreign::S>{{1}} == Definition::C<Foreign::S>{{1}});
static_assert(!(Definition::C<Foreign::S>{{1}} == Definition::C<Foreign::S>{{2}}));
} // namespace Caller

namespace TooLate {
template <class T> struct C {
  T m; // expected-note {{no viable 'operator==' for member 'm'}}
  bool operator==(const C &) const;
};
template <class T>
bool C<T>::operator==(const C &) const = default; // expected-error {{would delete it after its first declaration}}

// Ordinary lookup must not gain candidates declared after the definition,
// even though the saved lookup set was empty.
bool operator==(Foreign::S, Foreign::S);
bool use(C<Foreign::S> a, C<Foreign::S> b) {
  return a == b; // expected-note {{in instantiation of member function}}
}
} // namespace TooLate

namespace DeletedMember {
template <class T> struct C {
  T m; // expected-note {{no viable 'operator==' for member 'm'}}
  bool operator==(const C &) const;
};
template <class T>
bool C<T>::operator==(const C &) const = default; // expected-error {{would delete it after its first declaration}}

bool use(C<Foreign::Uncomparable> a, C<Foreign::Uncomparable> b) {
  return a == b; // expected-note {{in instantiation of member function}}
}
// An empty saved lookup set is also valid for builtin comparisons.
bool use(C<int> a, C<int> b) { return a == b; }
} // namespace DeletedMember

namespace InlineTiming {
template <class T> struct C {
  T m; // expected-note {{no viable 'operator==' for member 'm'}}
  bool operator==(const C &) const = default; // expected-note {{explicitly defaulted function was implicitly deleted here}}
};
bool operator==(Foreign::S, Foreign::S);
bool use(C<Foreign::S> a, C<Foreign::S> b) {
  return a == b; // expected-error {{because its 'operator==' is implicitly deleted}}
}
} // namespace InlineTiming

namespace Nested {
constexpr bool operator==(Foreign::S a, Foreign::S b) {
  return a.value == b.value;
}
template <class T> struct Outer {
  template <class U> struct Inner {
    Foreign::S m;
    constexpr bool operator==(const Inner &) const;
  };
};
template <class T>
template <class U>
constexpr bool Outer<T>::Inner<U>::operator==(const Inner &) const = default;

// Force partial instantiation before instantiating the inner comparison.
template struct Outer<int>;
using C = Outer<int>::Inner<float>;
static_assert(C{{1}} == C{{1}});
static_assert(!(C{{1}} == C{{2}}));
} // namespace Nested

namespace NestedTooLate {
template <class T> struct Outer {
  template <class U> struct Inner {
    Foreign::S m; // expected-note {{no viable 'operator==' for member 'm'}}
    bool operator==(const Inner &) const;
  };
};
template <class T>
template <class U>
bool Outer<T>::Inner<U>::operator==(const Inner &) const = default; // expected-error {{would delete it after its first declaration}}
bool operator==(Foreign::S, Foreign::S);
template struct Outer<int>;
bool use(Outer<int>::Inner<float> a, Outer<int>::Inner<float> b) {
  return a == b; // expected-note {{in instantiation of member function}}
}
} // namespace NestedTooLate

namespace Access {
template <class T> struct C;
struct M {
  int value;

private:
  constexpr bool operator==(const M &rhs) const {
    return value == rhs.value;
  }
  template <class T> friend struct C;
};
template <class T> struct C {
  M m;
  constexpr bool operator==(const C &) const;
};
template <class T>
constexpr bool C<T>::operator==(const C &) const = default;
static_assert(C<int>{{1}} == C<int>{{1}});
static_assert(!(C<int>{{1}} == C<int>{{2}}));
} // namespace Access

namespace LaterADL {
struct S {
  int value;
};
} // namespace LaterADL
namespace ADL {
template <class T> struct C {
  T m;
  constexpr bool operator==(const C &) const;
};
template <class T>
constexpr bool C<T>::operator==(const C &) const = default;
} // namespace ADL
namespace LaterADL {
// Unlike ordinary lookup, ADL can find this at instantiation.
constexpr bool operator==(S a, S b) { return a.value == b.value; }
} // namespace LaterADL
static_assert(ADL::C<LaterADL::S>{{1}} == ADL::C<LaterADL::S>{{1}});
static_assert(!(ADL::C<LaterADL::S>{{1}} == ADL::C<LaterADL::S>{{2}}));

namespace ImplicitEquality {
template <class T> struct Value {
  T value;
};
} // namespace ImplicitEquality
namespace Spaceship {
template <class T>
constexpr bool operator==(ImplicitEquality::Value<T> a,
                          ImplicitEquality::Value<T> b) {
  return a.value == b.value;
}
template <class T>
constexpr std::strong_ordering operator<=>(ImplicitEquality::Value<T> a,
                                           ImplicitEquality::Value<T> b) {
  return a.value <=> b.value;
}
template <class T> struct C {
  ImplicitEquality::Value<T> m;
  constexpr auto operator<=>(const C &) const = default;
};
static_assert(C<int>{{1}} == C<int>{{1}});
static_assert(!(C<int>{{1}} == C<int>{{2}}));
static_assert((C<int>{{1}} <=> C<int>{{1}}) == 0);
static_assert((C<int>{{1}} <=> C<int>{{2}}) < 0);
} // namespace Spaceship

namespace OutOfLineSpaceship {
template <class T> struct C {
  T m;
  constexpr std::strong_ordering operator<=>(const C &) const;
};
constexpr bool operator==(Foreign::S a, Foreign::S b) {
  return a.value == b.value;
}
constexpr bool operator<(Foreign::S a, Foreign::S b) {
  return a.value < b.value;
}
template <class T>
constexpr std::strong_ordering C<T>::operator<=>(const C &) const = default;
static_assert((C<Foreign::S>{{1}} <=> C<Foreign::S>{{1}}) == 0);
static_assert((C<Foreign::S>{{1}} <=> C<Foreign::S>{{2}}) < 0);
static_assert((C<Foreign::S>{{2}} <=> C<Foreign::S>{{1}}) > 0);
} // namespace OutOfLineSpaceship
