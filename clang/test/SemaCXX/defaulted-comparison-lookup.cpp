// RUN: %clang_cc1 -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++23 -verify %s
// RUN: %clang_cc1 -std=c++2c -verify %s

#include "Inputs/std-compare.h"

// GH75083: out-of-class defaulting must perform the ordinary operator lookup
// that would be performed in the function body, not just ADL.
namespace Foreign {
struct S {
  int value;
};
template <class T, int> struct CT {
  T value;
};
} // namespace Foreign

namespace Member {
struct C {
  Foreign::S s;
  constexpr bool operator==(const C &) const;
};

// Deliberately declared after C, but before the defaulted definition.
constexpr bool operator==(Foreign::S a, Foreign::S b) {
  return a.value == b.value;
}
constexpr bool C::operator==(const C &) const = default;
static_assert(C{{1}} == C{{1}});
static_assert(!(C{{1}} == C{{2}}));

struct Inline {
  Foreign::S s;
  constexpr bool operator==(const Inline &) const = default;
};
static_assert(Inline{{1}} == Inline{{1}});
static_assert(!(Inline{{1}} == Inline{{2}}));
} // namespace Member

namespace Friend {
struct C {
  Foreign::S s;
  friend constexpr bool operator==(const C &, const C &);
};
constexpr bool operator==(Foreign::S a, Foreign::S b) {
  return a.value == b.value;
}
constexpr bool operator==(const C &, const C &) = default;
static_assert(C{{1}} == C{{1}});
static_assert(!(C{{1}} == C{{2}}));

struct ByValue {
  Foreign::S s;
  friend constexpr bool operator==(ByValue, ByValue);
};
constexpr bool operator==(ByValue, ByValue) = default;
static_assert(ByValue{{1}} == ByValue{{1}});
static_assert(!(ByValue{{1}} == ByValue{{2}}));

struct Manual {
  Foreign::S s;
  friend constexpr bool operator==(const Manual &a, const Manual &b) {
    return a.s == b.s;
  }
};
static_assert(Manual{{1}} == Manual{{1}});
static_assert(!(Manual{{1}} == Manual{{2}}));
} // namespace Friend

namespace OperatorTemplate {
using Alias = Foreign::CT<int, 2>;
struct C {
  Alias a;
  Foreign::CT<float, 2> b;
  friend constexpr bool operator==(const C &, const C &);
};
template <class T>
constexpr bool operator==(Foreign::CT<T, 2> a, Foreign::CT<T, 2> b) {
  return a.value == b.value;
}
constexpr bool operator==(const C &, const C &) = default;
static_assert(C{{1}, {2.f}} == C{{1}, {2.f}});
static_assert(!(C{{1}, {2.f}} == C{{2}, {2.f}}));
static_assert(!(C{{1}, {2.f}} == C{{1}, {3.f}}));
} // namespace OperatorTemplate

namespace Qualified {
struct C {
  Foreign::S s;
  constexpr bool operator==(const C &) const;
};
constexpr bool operator==(Foreign::S a, Foreign::S b) {
  return a.value == b.value;
}
struct Manual {
  Foreign::S s;
  constexpr bool equal(const Manual &) const;
};
} // namespace Qualified

// The lexical namespace here is not the member function's semantic namespace.
constexpr bool Qualified::C::operator==(const C &) const = default;
constexpr bool Qualified::Manual::equal(const Manual &rhs) const {
  return s == rhs.s;
}
static_assert(Qualified::C{{1}} == Qualified::C{{1}});
static_assert(!(Qualified::C{{1}} == Qualified::C{{2}}));
static_assert(Qualified::Manual{{1}}.equal(Qualified::Manual{{1}}));

namespace Imported {
constexpr bool operator==(Foreign::S a, Foreign::S b) {
  return a.value == b.value;
}
} // namespace Imported

namespace UsingDirective {
using namespace Imported;
struct C {
  Foreign::S s;
  constexpr bool operator==(const C &) const;
};
constexpr bool C::operator==(const C &) const = default;
static_assert(C{{1}} == C{{1}});
static_assert(!(C{{1}} == C{{2}}));
} // namespace UsingDirective

namespace UsingDeclaration {
using Imported::operator==;
struct C {
  Foreign::S s;
  friend constexpr bool operator==(const C &, const C &);
};
constexpr bool operator==(const C &, const C &) = default;
static_assert(C{{1}} == C{{1}});
static_assert(!(C{{1}} == C{{2}}));
} // namespace UsingDeclaration

namespace Hiding {
bool operator==(Foreign::S, Foreign::S);
namespace Inner {
struct Other {};
bool operator==(Other, Other); // expected-note 2{{candidate function not viable}}
struct C {
  Foreign::S s; // expected-note {{no viable 'operator==' for member 's'}}
  bool operator==(const C &) const;
};
bool C::operator==(const C &) const = default; // expected-error {{would delete it after its first declaration}}

struct Manual {
  Foreign::S s;
  bool equal(const Manual &rhs) const {
    return s == rhs.s; // expected-error {{invalid operands to binary expression}}
  }
};
} // namespace Inner
} // namespace Hiding

namespace Associated {
struct S {
  int value;
};
} // namespace Associated
namespace ADL {
struct C {
  Associated::S s;
  friend constexpr bool operator==(const C &, const C &);
};
} // namespace ADL
namespace Associated {
constexpr bool operator==(S a, S b) { return a.value == b.value; }
} // namespace Associated
namespace ADL {
constexpr bool operator==(const C &, const C &) = default;
static_assert(C{{1}} == C{{1}});
static_assert(!(C{{1}} == C{{2}}));
} // namespace ADL

namespace DirectThreeWay {
struct C {
  Foreign::S s;
  constexpr std::strong_ordering operator<=>(const C &) const;
};
constexpr std::strong_ordering operator<=>(Foreign::S a, Foreign::S b) {
  return a.value <=> b.value;
}
constexpr std::strong_ordering C::operator<=>(const C &) const = default;
static_assert((C{{1}} <=> C{{1}}) == 0);
static_assert((C{{1}} <=> C{{2}}) < 0);
static_assert((C{{2}} <=> C{{1}}) > 0);
} // namespace DirectThreeWay

namespace ThreeWay {
struct C {
  Foreign::S s;
  constexpr std::strong_ordering operator<=>(const C &) const;
};
constexpr bool operator==(Foreign::S a, Foreign::S b) {
  return a.value == b.value;
}
constexpr bool operator<(Foreign::S a, Foreign::S b) {
  return a.value < b.value;
}
// Explicit-category three-way comparison falls back to ordinary == and <.
constexpr std::strong_ordering C::operator<=>(const C &) const = default;
static_assert((C{{1}} <=> C{{1}}) == 0);
static_assert((C{{1}} <=> C{{2}}) < 0);
static_assert((C{{2}} <=> C{{1}}) > 0);
} // namespace ThreeWay

namespace FPFeatures {
#pragma clang fp contract(fast)
struct C {
  Foreign::S s;
  constexpr bool operator==(const C &) const;
};
constexpr bool operator==(Foreign::S a, Foreign::S b) {
  return a.value == b.value;
}
constexpr bool C::operator==(const C &) const = default;
static_assert(C{{1}} == C{{1}});
static_assert(!(C{{1}} == C{{2}}));
} // namespace FPFeatures
