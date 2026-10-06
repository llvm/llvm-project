// RUN: %clang_cc1 -std=c++2d -verify %s

// C++2d [over.inc.default]p3:
//   The implicit definition of a defaulted postfix increment or decrement
//   operator function F that is not defined as deleted for a type C is
//   equivalent to:
//     C tmp(c);
//     ++c;
//     return tmp;
//   for a postfix increment operator function, or
//     C tmp(c);
//     --c;
//     return tmp;
//   for a postfix decrement operator function, where tmp is a variable defined
//   for exposition only, and c is an lvalue that denotes *this if F is an
//   implicit object member function, or the first parameter of F otherwise.

struct Counter {
  int v;
  constexpr Counter &operator++() { ++v; return *this; }
  constexpr Counter &operator--() { --v; return *this; }
  Counter operator++(int) = default;
  Counter operator--(int) = default;
};
constexpr int counter() {
  Counter c{1};
  Counter a = c++;
  Counter b = c--;
  return a.v * 100 + b.v * 10 + c.v;
}
static_assert(counter() == 121);

struct NonMember {
  int v;
  constexpr NonMember &operator++() { ++v; return *this; }
};
constexpr NonMember operator++(NonMember &, int) = default;
constexpr int non_member() {
  NonMember n{1};
  NonMember a = n++;
  return a.v * 10 + n.v;
}
static_assert(non_member() == 12);

struct ExplicitObject {
  int v;
  constexpr ExplicitObject &operator++() { ++v; return *this; }
  ExplicitObject operator++(this ExplicitObject &self, int) = default;
};
constexpr int explicit_object() {
  ExplicitObject e{1};
  ExplicitObject a = e++;
  return a.v * 10 + e.v;
}
static_assert(explicit_object() == 12);

struct RvalueObject {
  int v;
  constexpr RvalueObject &operator++() { ++v; return *this; }
  RvalueObject operator++(int) && = default;
};
constexpr int rvalue_object() {
  RvalueObject r{1};
  RvalueObject a = static_cast<RvalueObject &&>(r)++;
  return a.v * 10 + r.v;
}
static_assert(rvalue_object() == 12);

// The result of the prefix operator is discarded, so its type is irrelevant.
struct ReturnsVoid {
  int v;
  constexpr void operator++() { ++v; }
  ReturnsVoid operator++(int) = default;
};
constexpr int returns_void() {
  ReturnsVoid r{1};
  ReturnsVoid a = r++;
  return a.v * 10 + r.v;
}
static_assert(returns_void() == 12);

// Enumerations.
enum E { e0, e1, e2 };
constexpr E &operator++(E &e) { return e = E(e + 1); }
constexpr E operator++(E &, int) = default;
constexpr int enumeration() {
  E e = e0;
  E a = e++;
  return a * 10 + e;
}
static_assert(enumeration() == 1);

// The prefix operator may be a built-in operator reached through a conversion
// function.
struct ConvertsToInt {
  int v;
  constexpr operator int &() { return v; }
  ConvertsToInt operator++(int) = default;
};
constexpr int converts_to_int() {
  ConvertsToInt c{1};
  ConvertsToInt a = c++;
  return a.v * 10 + c.v;
}
static_assert(converts_to_int() == 12);

// Class templates.
template <typename T> struct Wrapper {
  T v;
  constexpr Wrapper &operator++() { ++v; return *this; }
  constexpr Wrapper &operator--() { --v; return *this; }
  Wrapper operator++(int) = default;
  friend Wrapper operator--(Wrapper &, int) = default;
};
constexpr int wrapper() {
  Wrapper<int> w{5};
  Wrapper<int> a = w++;
  Wrapper<int> b = w--;
  return a.v * 100 + b.v * 10 + w.v;
}
static_assert(wrapper() == 565);

// Direct-initialization of the copy permits an explicit copy constructor, but
// the return statement needs a usable move or copy constructor.
struct ExplicitCopy {
  explicit ExplicitCopy(const ExplicitCopy &);
  ExplicitCopy(ExplicitCopy &&);
  ExplicitCopy &operator++();
  ExplicitCopy operator++(int) = default; // OK
};
void explicit_copy(ExplicitCopy e) { e++; }

struct ExplicitCopyOnly {
  explicit ExplicitCopyOnly(const ExplicitCopyOnly &); // expected-note {{explicit constructor is not a candidate}}
  ExplicitCopyOnly &operator++();
  ExplicitCopyOnly operator++(int) = default; // expected-error {{no matching constructor for initialization of 'ExplicitCopyOnly'}}
};
void explicit_copy_only(ExplicitCopyOnly e) {
  e++; // expected-note {{in defaulted postfix increment operator for 'ExplicitCopyOnly' first required here}}
}

struct DeletedMove {
  DeletedMove(const DeletedMove &);
  DeletedMove(DeletedMove &&) = delete; // expected-note {{'DeletedMove' has been explicitly marked deleted here}}
  DeletedMove &operator++();
  DeletedMove operator++(int) = default; // expected-error {{call to deleted constructor of 'DeletedMove'}}
};
void deleted_move(DeletedMove d) {
  d++; // expected-note {{in defaulted postfix increment operator for 'DeletedMove' first required here}}
}

// Attributes on the prefix operator are honored, as for any other call.
struct Nodiscard {
  [[nodiscard]] Nodiscard &operator++();
  Nodiscard operator++(int) = default; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
};
void nodiscard(Nodiscard n) {
  n++; // expected-note {{in defaulted postfix increment operator for 'Nodiscard' first required here}}
}

// Volatile objects.
struct Volatile {
  Volatile();
  Volatile(const Volatile &);
  Volatile(const volatile Volatile &);
  Volatile(Volatile &&);
  Volatile &operator++() volatile;
  Volatile operator++(int) volatile = default;
};
void use_volatile() {
  volatile Volatile v;
  v++;
}

// Name lookup for the prefix operator is performed from a context equivalent
// to the function body, even if the body is synthesized elsewhere.
namespace Lookup {
namespace A {
struct X { int v; };
struct Y;
} // namespace A
namespace B {
constexpr A::Y &operator++(A::Y &y);
} // namespace B
namespace C {
namespace Nested {
constexpr A::X &operator++(A::X &x) { ++x.v; return x; }
} // namespace Nested
using namespace Nested;
// Found by unqualified lookup through the using-directive, not by ADL.
constexpr A::X operator++(A::X &, int) = default;
} // namespace C
constexpr int non_member() {
  A::X x{1};
  A::X a = C::operator++(x, 0);
  return a.v * 10 + x.v;
}
static_assert(non_member() == 12);

namespace A {
using namespace B;
struct Y {
  int v;
  Y operator++(int) = default;
};
} // namespace A
constexpr A::Y &B::operator++(A::Y &y) { ++y.v; return y; }
namespace D {
constexpr int member() {
  A::Y y{1};
  A::Y a = y++; // The body is synthesized here, where B::operator++ is not visible.
  return a.v * 10 + y.v;
}
static_assert(member() == 12);
} // namespace D
} // namespace Lookup

// C++2d [except.spec]p10:
//   The exception specification for [...] a postfix increment or decrement
//   operator function without a noexcept-specifier that is defaulted on its
//   first declaration is potentially-throwing if and only if any expression in
//   the implicit definition is potentially-throwing.
struct Noexcept {
  Noexcept &operator++() noexcept;
  Noexcept operator++(int) = default;
};
static_assert(noexcept(Noexcept{}++));
struct ThrowingPrefix {
  ThrowingPrefix &operator++();
  ThrowingPrefix operator++(int) = default;
};
static_assert(!noexcept(ThrowingPrefix{}++));
struct ThrowingCopy {
  ThrowingCopy();
  ThrowingCopy(const ThrowingCopy &) noexcept(false);
  ThrowingCopy &operator++() noexcept;
  ThrowingCopy operator++(int) = default;
};
static_assert(!noexcept(ThrowingCopy{}++));
struct ThrowingDestructor {
  ThrowingDestructor();
  ~ThrowingDestructor() noexcept(false);
  ThrowingDestructor &operator++() noexcept;
  ThrowingDestructor operator++(int) = default;
};
static_assert(!noexcept(ThrowingDestructor{}++));
struct ExplicitNoexcept {
  ExplicitNoexcept &operator++();
  ExplicitNoexcept operator++(int) noexcept = default;
};
static_assert(noexcept(ExplicitNoexcept{}++));
struct ThrowingNonMember {
  ThrowingNonMember &operator++();
};
ThrowingNonMember operator++(ThrowingNonMember &, int) = default;
extern ThrowingNonMember throwing_non_member;
static_assert(!noexcept(throwing_non_member++));
struct NoexceptNonMember {
  NoexceptNonMember &operator++() noexcept;
};
NoexceptNonMember operator++(NoexceptNonMember &, int) = default;
extern NoexceptNonMember noexcept_non_member;
static_assert(noexcept(noexcept_non_member++));
