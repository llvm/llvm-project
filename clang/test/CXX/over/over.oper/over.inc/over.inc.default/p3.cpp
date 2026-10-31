// RUN: %clang_cc1 -std=c++2d -verify %s
// RUN: %clang_cc1 -std=c++2d -funknown-anytype -DUNKNOWN_ANYTYPE -verify=anytype %s

#ifdef UNKNOWN_ANYTYPE
// The discarded-value prefix call must have a known type.
struct UnknownPrefixResult {
  __unknown_anytype operator++();
  UnknownPrefixResult operator++(int) = default; // anytype-error {{'operator++' has unknown return type; cast the call to its declared return type}}
};
void unknown_prefix_result(UnknownPrefixResult u) {
  u++; // anytype-note {{in defaulted postfix increment operator for 'UnknownPrefixResult' first required here}}
}
#else

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

// [over.inc.default]p2 checks the copy from an lvalue of type C, but in the
// implicit definition of a volatile member, c is an lvalue of type volatile C.
struct VolatileAggregate {
  volatile VolatileAggregate &operator++() volatile;
  VolatileAggregate operator++(int) volatile = default; // expected-error {{excess elements in struct initializer}}
};
void volatile_aggregate(VolatileAggregate v) {
  v++; // expected-note {{in defaulted postfix increment operator for 'VolatileAggregate' first required here}}
}

// Errors that overload resolution does not detect make the implicit definition
// ill-formed rather than deleted.
struct BadDefaultArgument {
  BadDefaultArgument();
  template <class T> BadDefaultArgument(T &, int = T::missing); // expected-error 2 {{no member named 'missing' in 'BadDefaultArgument'}}
  BadDefaultArgument &operator++();
  BadDefaultArgument operator++(int) = default; // expected-note 2 {{in instantiation of default function argument expression for 'BadDefaultArgument<BadDefaultArgument>' required here}}
};
void bad_default_argument(BadDefaultArgument b) {
  b++; // expected-note 2 {{in defaulted postfix increment operator for 'BadDefaultArgument' first required here}}
}

struct Incomplete; // expected-note 2 {{forward declaration of 'Incomplete'}}
struct IncompletePrefixResult {
  Incomplete operator++(); // expected-note {{'operator++' declared here}}
  IncompletePrefixResult operator++(int) = default; // expected-error {{calling 'operator++' with incomplete return type 'Incomplete'}}
};
void incomplete_prefix_result(IncompletePrefixResult i) {
  i++; // expected-note {{in defaulted postfix increment operator for 'IncompletePrefixResult' first required here}}
}

struct DeletedDestructor {
  ~DeletedDestructor() = delete; // expected-note {{'~DeletedDestructor' has been explicitly marked deleted here}}
};
struct DeletedDestructorPrefixResult {
  DeletedDestructor operator--();
  DeletedDestructorPrefixResult operator--(int) = default; // expected-error {{attempt to use a deleted function}}
};
void deleted_destructor_prefix_result(DeletedDestructorPrefixResult d) {
  d--; // expected-note {{in defaulted postfix decrement operator for 'DeletedDestructorPrefixResult' first required here}}
}

// Uses of the first parameter in the implicit definition are checked as usual.
struct UnavailableParam {
  UnavailableParam &operator++();
};
UnavailableParam operator++(UnavailableParam &u __attribute__((unavailable)), int) = default; // expected-note 2 {{'u' has been explicitly marked unavailable here}}
// expected-error@-1 2 {{'u' is unavailable}}
// expected-note@-2 {{in defaulted postfix increment operator for 'UnavailableParam' first required here}}
void unavailable_param(UnavailableParam u) { u++; }

// The same errors are diagnosed when the body is only built to compute the
// exception specification.
struct IncompletePrefixResultNoexcept {
  Incomplete operator++(); // expected-note {{'operator++' declared here}}
  IncompletePrefixResultNoexcept operator++(int) = default; // expected-error {{calling 'operator++' with incomplete return type 'Incomplete'}}
};
extern IncompletePrefixResultNoexcept incomplete_prefix_result_noexcept;
bool incomplete_noexcept = noexcept(incomplete_prefix_result_noexcept++); // expected-note {{in evaluation of exception specification for 'IncompletePrefixResultNoexcept::operator++' needed here}}

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
struct ThrowingDestructorDefined {
  ThrowingDestructorDefined();
  ~ThrowingDestructorDefined() noexcept(false);
  ThrowingDestructorDefined &operator++() noexcept;
  ThrowingDestructorDefined operator++(int) = default;
};
void throwing_destructor_defined(ThrowingDestructorDefined t) { t++; }
static_assert(!noexcept(ThrowingDestructorDefined{}++));
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

// Prefix operators with an explicit object parameter.
struct ExplicitObjectPrefix {
  int v;
  constexpr ExplicitObjectPrefix &operator++(this ExplicitObjectPrefix &self) { ++self.v; return self; }
  constexpr ExplicitObjectPrefix &operator--(this ExplicitObjectPrefix &self) { --self.v; return self; }
  ExplicitObjectPrefix operator++(int) = default;
  friend ExplicitObjectPrefix operator--(ExplicitObjectPrefix &, int) = default;
};
constexpr int explicit_object_prefix() {
  ExplicitObjectPrefix e{1};
  ExplicitObjectPrefix a = e++;
  ExplicitObjectPrefix b = e--;
  return a.v * 100 + b.v * 10 + e.v;
}
static_assert(explicit_object_prefix() == 121);

// A deduced explicit object parameter deduces an lvalue of type C.
struct DeducedObjectPrefix {
  int v;
  constexpr auto &operator++(this auto &&self) {
    static_assert(__is_same(decltype(self), DeducedObjectPrefix &));
    ++self.v;
    return self;
  }
  DeducedObjectPrefix operator++(int) = default;
};
constexpr int deduced_object_prefix() {
  DeducedObjectPrefix d{1};
  DeducedObjectPrefix a = d++;
  return a.v * 10 + d.v;
}
static_assert(deduced_object_prefix() == 12);

// The explicit object parameter need not be a non-const lvalue reference; the
// prefix operator is called like any other.
struct ByValueObjectPrefix {
  int v;
  constexpr ByValueObjectPrefix operator++(this ByValueObjectPrefix self) { return {self.v + 1}; }
  ByValueObjectPrefix operator++(int) = default;
};
constexpr int by_value_object_prefix() {
  ByValueObjectPrefix b{1};
  ByValueObjectPrefix a = b++;
  return a.v * 10 + b.v;
}
static_assert(by_value_object_prefix() == 11);
struct ConstObjectPrefix {
  mutable int v;
  constexpr const ConstObjectPrefix &operator--(this const ConstObjectPrefix &self) { --self.v; return self; }
  ConstObjectPrefix operator--(int) = default;
};
constexpr int const_object_prefix() {
  ConstObjectPrefix c{1};
  ConstObjectPrefix a = c--;
  return a.v * 10 + c.v;
}
static_assert(const_object_prefix() == 10);

// Postfix operators with an explicit object parameter.
struct ExplicitObjectRvalue {
  int v;
  constexpr ExplicitObjectRvalue &operator++() { ++v; return *this; }
  ExplicitObjectRvalue operator++(this ExplicitObjectRvalue &&, int) = default;
};
constexpr int explicit_object_rvalue() {
  ExplicitObjectRvalue e{1};
  ExplicitObjectRvalue a = static_cast<ExplicitObjectRvalue &&>(e)++;
  return a.v * 10 + e.v;
}
static_assert(explicit_object_rvalue() == 12);
struct ExplicitObjectVolatile {
  ExplicitObjectVolatile();
  ExplicitObjectVolatile(const volatile ExplicitObjectVolatile &);
  ExplicitObjectVolatile(ExplicitObjectVolatile &&);
  ExplicitObjectVolatile &operator++(this volatile ExplicitObjectVolatile &);
  ExplicitObjectVolatile operator++(this volatile ExplicitObjectVolatile &, int) = default;
};
void explicit_object_volatile() {
  volatile ExplicitObjectVolatile v;
  v++;
}
// Both operators have an explicit object parameter.
template <typename T> struct ExplicitObjectBoth {
  T v;
  constexpr ExplicitObjectBoth &operator++(this ExplicitObjectBoth &self) { ++self.v; return self; }
  constexpr ExplicitObjectBoth &operator--(this auto &self) { --self.v; return self; }
  ExplicitObjectBoth operator++(this ExplicitObjectBoth &self, int) = default;
  ExplicitObjectBoth operator--(this ExplicitObjectBoth &self, int) = default;
};
constexpr int explicit_object_both() {
  ExplicitObjectBoth<int> e{1};
  ExplicitObjectBoth<int> a = e++;
  ExplicitObjectBoth<int> b = e--;
  return a.v * 100 + b.v * 10 + e.v;
}
static_assert(explicit_object_both() == 121);
// An explicit object postfix operator with an implicit object prefix operator
// is called the same way.
struct ExplicitObjectPostfixOnly {
  int v;
  constexpr ExplicitObjectPostfixOnly &operator--() { --v; return *this; }
  ExplicitObjectPostfixOnly operator--(this ExplicitObjectPostfixOnly &self, int) = default;
};
constexpr int explicit_object_postfix_only() {
  ExplicitObjectPostfixOnly e{1};
  ExplicitObjectPostfixOnly a = e--;
  return a.v * 10 + e.v;
}
static_assert(explicit_object_postfix_only() == 10);

// Constrained prefix operators.
template <typename T> concept Small = sizeof(T) <= 4;
template <typename T> concept Int = __is_same(T, int);
template <typename T, typename U> concept Same = __is_same(T, U);
template <typename T> struct ConstrainedPrefix {
  T v;
  constexpr ConstrainedPrefix &operator++() requires Small<T> { ++v; return *this; }
  constexpr ConstrainedPrefix &operator--(this ConstrainedPrefix &self) requires Small<T> { --self.v; return self; }
  ConstrainedPrefix operator++(int) = default;
  ConstrainedPrefix operator--(int) = default;
};
constexpr int constrained_prefix() {
  ConstrainedPrefix<int> c{1};
  ConstrainedPrefix<int> a = c++;
  ConstrainedPrefix<int> b = c--;
  return a.v * 100 + b.v * 10 + c.v;
}
static_assert(constrained_prefix() == 121);
// The more constrained prefix operator is selected.
template <typename T> struct MoreConstrainedPrefix {
  T v;
  constexpr MoreConstrainedPrefix &operator++() requires Small<T> { v += 1; return *this; }
  constexpr MoreConstrainedPrefix &operator++() requires Small<T> && Int<T> { v += 10; return *this; }
  MoreConstrainedPrefix operator++(int) = default;
};
constexpr int more_constrained_prefix() {
  MoreConstrainedPrefix<int> i{1};
  MoreConstrainedPrefix<int> a = i++;
  MoreConstrainedPrefix<short> s{1};
  MoreConstrainedPrefix<short> b = s++;
  return a.v * 1000 + i.v * 100 + b.v * 10 + s.v;
}
static_assert(more_constrained_prefix() == 1 * 1000 + 11 * 100 + 1 * 10 + 2);
// A constrained non-member prefix operator template.
struct ConstrainedNonMemberPrefix { int v; };
template <typename T> requires __is_same(T, ConstrainedNonMemberPrefix)
constexpr T &operator++(T &t) { ++t.v; return t; }
constexpr ConstrainedNonMemberPrefix operator++(ConstrainedNonMemberPrefix &, int) = default;
constexpr int constrained_non_member_prefix() {
  ConstrainedNonMemberPrefix c{1};
  ConstrainedNonMemberPrefix a = c++;
  return a.v * 10 + c.v;
}
static_assert(constrained_non_member_prefix() == 12);
// The exception specification depends on the selected prefix operator.
template <typename T> struct ConstrainedPrefixNoexcept {
  T v;
  ConstrainedPrefixNoexcept &operator++() noexcept requires Small<T>;
  ConstrainedPrefixNoexcept &operator++();
  ConstrainedPrefixNoexcept operator++(int) = default;
};
static_assert(noexcept(ConstrainedPrefixNoexcept<int>{}++));
static_assert(!noexcept(ConstrainedPrefixNoexcept<long long>{}++));

// Constrained postfix operators.
template <typename T> struct ConstrainedPostfix {
  T v;
  constexpr ConstrainedPostfix &operator++() { ++v; return *this; }
  constexpr ConstrainedPostfix &operator--() { --v; return *this; }
  ConstrainedPostfix operator++(int) requires Small<T> = default;
  friend ConstrainedPostfix operator--(ConstrainedPostfix &, int) requires Small<T> = default;
};
constexpr int constrained_postfix() {
  ConstrainedPostfix<int> c{1};
  ConstrainedPostfix<int> a = c++;
  ConstrainedPostfix<int> b = c--;
  return a.v * 100 + b.v * 10 + c.v;
}
static_assert(constrained_postfix() == 121);
void constrained_postfix_unsatisfied(ConstrainedPostfix<long long> c) {
  c++; // expected-error {{cannot increment value of type 'ConstrainedPostfix<long long>'}}
  c--; // expected-error {{cannot decrement value of type 'ConstrainedPostfix<long long>'}}
}
// A defaulted postfix operator can be more constrained than a user-provided
// one, and vice versa.
template <typename T> struct MoreConstrainedPostfix {
  T v;
  constexpr MoreConstrainedPostfix &operator++() { v += 1; return *this; }
  constexpr MoreConstrainedPostfix operator++(int) requires Small<T> { MoreConstrainedPostfix t = *this; v += 10; return t; }
  MoreConstrainedPostfix operator++(int) requires Small<T> && Int<T> = default;
  constexpr MoreConstrainedPostfix &operator--() { v -= 1; return *this; }
  MoreConstrainedPostfix operator--(int) requires Small<T> = default;
  constexpr MoreConstrainedPostfix operator--(int) requires Small<T> && Int<T> { MoreConstrainedPostfix t = *this; v -= 10; return t; }
};
constexpr int more_constrained_postfix() {
  MoreConstrainedPostfix<int> i{1};
  MoreConstrainedPostfix<int> a = i++;
  MoreConstrainedPostfix<int> b = i--;
  MoreConstrainedPostfix<short> s{1};
  MoreConstrainedPostfix<short> c = s++;
  MoreConstrainedPostfix<short> d = s--;
  return a.v * 100000 + b.v * 10000 + i.v * 1000 + c.v * 100 + d.v * 10 + s.v;
}
static_assert(more_constrained_postfix() == 1 * 100000 + 2 * 10000 + (-8) * 1000 + 1 * 100 + 11 * 10 + 10);
// A constrained explicit object postfix operator.
template <typename T> struct ConstrainedExplicitObjectPostfix {
  T v;
  constexpr ConstrainedExplicitObjectPostfix &operator++(this ConstrainedExplicitObjectPostfix &self) requires Small<T> { ++self.v; return self; }
  ConstrainedExplicitObjectPostfix operator++(this ConstrainedExplicitObjectPostfix &self, int) requires Small<T> = default;
};
constexpr int constrained_explicit_object_postfix() {
  ConstrainedExplicitObjectPostfix<int> c{1};
  ConstrainedExplicitObjectPostfix<int> a = c++;
  return a.v * 10 + c.v;
}
static_assert(constrained_explicit_object_postfix() == 12);
void constrained_explicit_object_postfix_unsatisfied(ConstrainedExplicitObjectPostfix<long long> c) {
  c++; // expected-error {{cannot increment value of type 'ConstrainedExplicitObjectPostfix<long long>'}}
}
// The exception specification of an explicit object postfix operator.
struct ExplicitObjectNoexcept {
  ExplicitObjectNoexcept &operator++(this ExplicitObjectNoexcept &) noexcept;
  ExplicitObjectNoexcept operator++(this ExplicitObjectNoexcept &, int) = default;
  ExplicitObjectNoexcept &operator--(this ExplicitObjectNoexcept &);
  ExplicitObjectNoexcept operator--(this ExplicitObjectNoexcept &, int) = default;
};
extern ExplicitObjectNoexcept explicit_object_noexcept;
static_assert(noexcept(explicit_object_noexcept++));
static_assert(!noexcept(explicit_object_noexcept--));
// A requires-expression observes whether a defaulted postfix operator is
// usable, its exception specification and its return type.
template <typename T> concept PostIncrementable = requires(T t) { t++; };
template <typename T> concept NoexceptPostIncrementable = requires(T t) {
  { t++ } noexcept -> Same<T>;
};
struct NoPrefix {
  NoPrefix operator++(int) = default; // #NoPrefix
  // expected-warning@#NoPrefix {{explicitly defaulted postfix increment operator is implicitly deleted}}
  // expected-note@#NoPrefix {{defaulted 'operator++' is implicitly deleted because there is no viable prefix 'operator++' for an lvalue of type 'NoPrefix'}}
  // expected-note@#NoPrefix {{replace 'default' with 'delete'}}
};
static_assert(PostIncrementable<Noexcept>);
static_assert(NoexceptPostIncrementable<Noexcept>);
static_assert(PostIncrementable<ThrowingPrefix>);
static_assert(!NoexceptPostIncrementable<ThrowingPrefix>);
static_assert(!PostIncrementable<NoPrefix>);
static_assert(PostIncrementable<ConstrainedPostfix<int>>);
static_assert(!PostIncrementable<ConstrainedPostfix<long long>>);
// A defaulted postfix operator constrained on its own prefix operator is not
// viable, rather than deleted, when there is no prefix operator.
template <typename T> struct PrefixConstrainedPostfix {
  T v;
  constexpr PrefixConstrainedPostfix &operator++() requires Small<T> { ++v; return *this; }
  PrefixConstrainedPostfix operator++(int) requires requires(PrefixConstrainedPostfix &p) { ++p; } = default;
};
constexpr int prefix_constrained_postfix() {
  PrefixConstrainedPostfix<int> p{1};
  PrefixConstrainedPostfix<int> a = p++;
  return a.v * 10 + p.v;
}
static_assert(prefix_constrained_postfix() == 12);
static_assert(PostIncrementable<PrefixConstrainedPostfix<int>>);
static_assert(!PostIncrementable<PrefixConstrainedPostfix<long long>>);
void prefix_constrained_postfix_unsatisfied(PrefixConstrainedPostfix<long long> p) {
  p++; // expected-error {{cannot increment value of type 'PrefixConstrainedPostfix<long long>'}}
}
#endif
