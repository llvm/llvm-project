// RUN: %clang_cc1 -std=c++2d -verify %s

// C++2d [over.inc.default]p1:
//   A defaulted postfix increment or decrement operator function for a type C
//   shall be a non-template function that
//    -- has a first parameter of type "reference to C" or a first parameter of
//       type "reference to volatile C", where the implicit object parameter
//       (if any) is considered to be the first parameter,
//    -- is defined as defaulted in C or in a context where C is complete, and
//    -- has a declared return type of C.
//   [...] A definition of a postfix increment or decrement operator as
//   defaulted that appears in a class shall be the first declaration of that
//   function.

struct S {
  S &operator++();
  S &operator--();

  S operator++(int) = default;
  S operator--(int) = default;
};

// The implicit object parameter is the first parameter.
struct Const {
  Const &operator++();
  Const operator++(int) const = default; // expected-error {{defaulted member postfix increment operator cannot be const-qualified}}
};
struct ConstDecrement {
  ConstDecrement &operator--();
  ConstDecrement operator--(int) const = default; // expected-error {{defaulted member postfix decrement operator cannot be const-qualified}}
};
struct Volatile {
  Volatile &operator++() volatile;
  Volatile operator++(int) volatile = default; // OK, "reference to volatile C"
};
struct RefQualified {
  RefQualified &operator++();
  RefQualified operator++(int) & = default;  // OK
  RefQualified operator++(int) && = default; // OK, an rvalue reference is a "reference to C"
};

// __restrict__ is not part of the type of the implicit object parameter.
struct Restrict {
  Restrict &operator++();
  Restrict operator++(int) __restrict__ = default; // OK
};
struct RestrictBoth {
  RestrictBoth &operator--() __restrict__;
  RestrictBoth operator--(int) __restrict__ = default; // OK
};
struct RestrictRvalue {
  RestrictRvalue &operator++();
  RestrictRvalue operator++(int) __restrict__ && = default; // OK
};
struct RestrictVolatile {
  RestrictVolatile &operator++() volatile;
  RestrictVolatile(const volatile RestrictVolatile &);
  RestrictVolatile(RestrictVolatile &&);
  RestrictVolatile operator++(int) volatile __restrict__ = default; // OK
};
struct RestrictConst {
  RestrictConst &operator++();
  RestrictConst operator++(int) const __restrict__ = default; // expected-error {{defaulted member postfix increment operator cannot be const-qualified}}
};
struct ExplicitRestrict {
  ExplicitRestrict &operator++();
  ExplicitRestrict operator++(this ExplicitRestrict &__restrict__, int) = default; // OK
};

// Explicit object parameters.
struct Explicit {
  Explicit &operator++();
  Explicit operator++(this Explicit &, int) = default; // OK
};
struct ExplicitVolatile {
  ExplicitVolatile &operator++() volatile;
  ExplicitVolatile operator++(this volatile ExplicitVolatile &, int) = default; // OK
};
struct ExplicitRvalue {
  ExplicitRvalue &operator++();
  ExplicitRvalue operator++(this ExplicitRvalue &&, int) = default; // OK
};
struct ExplicitConst {
  ExplicitConst &operator++();
  ExplicitConst operator++(this const ExplicitConst &, int) = default; // expected-error {{invalid explicit object parameter type for defaulted postfix increment operator; found 'const ExplicitConst &', expected 'ExplicitConst &' or 'volatile ExplicitConst &'}}
};
struct ExplicitOther {
  ExplicitOther &operator++();
  ExplicitOther operator++(this S &, int) = default; // expected-error {{invalid explicit object parameter type for defaulted postfix increment operator; found 'S &', expected 'ExplicitOther &' or 'volatile ExplicitOther &'}}
};
struct ExplicitByValue {
  ExplicitByValue &operator--();
  ExplicitByValue operator--(this ExplicitByValue, int) = default; // expected-error {{invalid explicit object parameter type for defaulted postfix decrement operator; found 'ExplicitByValue', expected 'ExplicitByValue &' or 'volatile ExplicitByValue &'}}
};

// Return type.
struct ReturnsRef {
  ReturnsRef &operator++();
  ReturnsRef &operator++(int) = default; // expected-error {{return type for defaulted postfix increment operator must be 'ReturnsRef', not 'ReturnsRef &'}}
};
struct ReturnsConst {
  ReturnsConst &operator++();
  const ReturnsConst operator++(int) = default; // expected-error {{return type for defaulted postfix increment operator must be 'ReturnsConst', not 'const ReturnsConst'}}
};
struct ReturnsAuto {
  ReturnsAuto &operator++();
  auto operator++(int) = default; // expected-error {{defaulted postfix increment operator cannot have a deduced return type}}
};
struct ReturnsDecltypeAuto {
  ReturnsDecltypeAuto &operator--();
  decltype(auto) operator--(int) = default; // expected-error {{defaulted postfix decrement operator cannot have a deduced return type}}
};
struct ReturnsInt {
  ReturnsInt &operator++();
  int operator++(int) = default; // expected-error {{return type for defaulted postfix increment operator must be 'ReturnsInt', not 'int'}}
};
struct ReturnsAlias {
  using Self = ReturnsAlias;
  Self &operator++();
  Self operator++(int) = default; // OK
};
struct ReturnsTrailing {
  ReturnsTrailing &operator++();
  auto operator++(int) -> ReturnsTrailing = default; // OK, not a deduced return type
};

// A placeholder type is not C, even if deduction would produce C.
template <typename T> concept Any = true;
struct ReturnsConstrainedAuto {
  ReturnsConstrainedAuto &operator++();
  Any auto operator++(int) = default; // expected-error {{defaulted postfix increment operator cannot have a deduced return type}}
};
struct ReturnsAutoPointer {
  ReturnsAutoPointer &operator++();
  auto *operator++(int) = default; // expected-error {{defaulted postfix increment operator cannot have a deduced return type}}
};
struct ReturnsAutoRef {
  ReturnsAutoRef &operator--();
  auto &operator--(int) = default; // expected-error {{defaulted postfix decrement operator cannot have a deduced return type}}
};
struct ReturnsConstAuto {
  ReturnsConstAuto &operator++();
  const auto operator++(int) = default; // expected-error {{defaulted postfix increment operator cannot have a deduced return type}}
};
struct ReturnsConstrainedAutoRef {
  ReturnsConstrainedAutoRef &operator++();
  const Any auto &operator++(int) = default; // expected-error {{defaulted postfix increment operator cannot have a deduced return type}}
};

// Non-member functions.
struct N {
  N &operator++();
  N &operator--();
};
N operator++(N &, int) = default;  // OK
N operator--(N &&, int) = default; // OK, an rvalue reference is a "reference to C"

struct NV {
  NV &operator++() volatile;
  NV(const volatile NV &);
  NV(NV &&);
};
NV operator++(volatile NV &, int) = default; // OK

struct NConst { NConst &operator++(); };
NConst operator++(const NConst &, int) = default; // expected-error {{invalid first parameter type for defaulted postfix increment operator; found 'const NConst &', expected 'NConst &' or 'volatile NConst &'}}
struct NValue { NValue &operator++(); };
NValue operator++(NValue, int) = default; // expected-error {{invalid first parameter type for defaulted non-member postfix increment operator; found 'NValue', expected reference to a non-const class or enumeration type}}
struct NRet { NRet &operator--(); };
int operator--(NRet &, int) = default; // expected-error {{return type for defaulted postfix decrement operator must be 'NRet', not 'int'}}
struct NMismatch { NMismatch &operator++(); };
S operator++(NMismatch &, int) = default; // expected-error {{return type for defaulted postfix increment operator must be 'NMismatch', not 'S'}}
struct NUnknown { NUnknown &operator++(); };
S &operator++(NUnknown, int) = default; // expected-error {{invalid first parameter type for defaulted non-member postfix increment operator; found 'NUnknown', expected reference to a non-const class or enumeration type}}
struct NAuto {
  NAuto &operator++();
  NAuto &operator--();
};
auto operator++(NAuto &, int) = default;           // expected-error {{defaulted postfix increment operator cannot have a deduced return type}}
Any auto operator--(NAuto &, int) = default;       // expected-error {{defaulted postfix decrement operator cannot have a deduced return type}}
decltype(auto) operator++(NAuto &&, int) = default; // expected-error {{defaulted postfix increment operator cannot have a deduced return type}}
auto operator--(NAuto &&, int) -> NAuto = default;  // OK
struct NRestrict { NRestrict &operator++(); };
NRestrict operator++(NRestrict &__restrict__, int) = default; // OK

// Enumerations.
enum E { e };
E &operator++(E &);
E operator++(E &, int) = default; // OK
enum class EC { ec };
EC operator++(EC &, int) = default; // OK, defined as deleted (see p2.cpp)
// expected-warning@-1 {{explicitly defaulted postfix increment operator is implicitly deleted}}
// expected-note@-2 {{defaulted 'operator++' is implicitly deleted because there is no viable prefix 'operator++' for an lvalue of type 'EC'}}
// expected-note@-3 {{replace 'default' with 'delete'}}

// C must be complete.
struct Incomplete; // expected-note {{forward declaration of 'Incomplete'}}
Incomplete operator++(Incomplete &, int) = default; // expected-error {{incomplete result type 'Incomplete' in function definition}}
struct IncompleteFriend; // expected-note {{forward declaration of 'IncompleteFriend'}}
struct Befriender {
  friend IncompleteFriend operator++(IncompleteFriend &, int) = default; // expected-error {{cannot default postfix increment operator for incomplete type 'IncompleteFriend'}}
};
struct DefinedLater {
  friend DefinedLater operator++(DefinedLater &, int) = default; // OK, defaulted in C
  DefinedLater &operator++();
};

// A definition as defaulted that appears in a class shall be the first
// declaration of that function.
struct First;
First operator++(First &, int); // expected-note {{previous declaration is here}}
struct First {
  First &operator++();
  friend First operator++(First &, int) = default; // expected-error {{defaulting this postfix increment operator is not allowed because it was already declared outside the class}}
};
struct OutOfLine {
  OutOfLine &operator++();
  OutOfLine operator++(int);
};
OutOfLine OutOfLine::operator++(int) = default; // OK
struct OutOfLineFriend {
  OutOfLineFriend &operator--();
  friend OutOfLineFriend operator--(OutOfLineFriend &, int);
};
OutOfLineFriend operator--(OutOfLineFriend &, int) = default; // OK
struct Redefined {
  Redefined &operator++();
  Redefined operator++(int) = default;
};
Redefined Redefined::operator++(int) { return *this; } // expected-error {{definition of explicitly defaulted function}}

// Templates cannot be defaulted.
struct Template {
  Template &operator++();
  template <typename T = void> Template operator++(int) = default; // expected-error {{postfix increment operator template cannot be defaulted}}
};
template <typename T = void> S operator--(S &, int) = default; // expected-error {{postfix decrement operator template cannot be defaulted}}

// A parameter declared with a placeholder type makes the function an
// abbreviated function template.
struct AutoFirst {
  AutoFirst &operator++();
  AutoFirst &operator--();
  AutoFirst operator++(this auto &, int) = default;   // expected-error {{postfix increment operator template cannot be defaulted}}
  friend AutoFirst operator--(auto &, int) = default; // expected-error {{postfix decrement operator template cannot be defaulted}}
};
AutoFirst operator++(Any auto &, int) = default; // expected-error {{postfix increment operator template cannot be defaulted}}
AutoFirst operator--(auto &&, int) = default;    // expected-error {{postfix decrement operator template cannot be defaulted}}

// The second parameter of a postfix operator always has type int.
struct AutoSecond {
  AutoSecond &operator++();
  AutoSecond &operator--();
  AutoSecond operator++(auto) = default;                    // expected-error {{postfix increment operator template cannot be defaulted}}
  AutoSecond operator--(this AutoSecond &, auto) = default; // expected-error {{postfix decrement operator template cannot be defaulted}}
};
AutoSecond operator++(AutoSecond &&, Any auto) = default; // expected-error {{postfix increment operator template cannot be defaulted}}
struct NotInt {
  NotInt &operator++();
  NotInt &operator--();
  NotInt operator++(long) = default;                    // expected-error {{parameter of overloaded post-increment operator must have type 'int' (not 'long')}}
  NotInt operator--(this NotInt &, unsigned) = default; // expected-error {{parameter of overloaded post-decrement operator must have type 'int' (not 'unsigned int')}}
};
NotInt operator++(NotInt &&, char) = default; // expected-error {{parameter of overloaded post-increment operator must have type 'int' (not 'char')}}
struct IntSpellings {
  using Int = int;
  IntSpellings &operator++();
  IntSpellings &operator--();
  IntSpellings operator++(const int) = default; // OK
  IntSpellings operator--(Int) = default;       // OK
};
struct DefaultArgument {
  DefaultArgument &operator++();
  DefaultArgument operator++(int = 0) = default; // expected-error {{parameter of overloaded 'operator++' cannot have a default argument}}
};

// Only the postfix forms can be defaulted.
struct Prefix {
  Prefix &operator++() = default; // expected-error {{only the postfix form of 'operator++' can be defaulted}}
  Prefix &operator--() = default; // expected-error {{only the postfix form of 'operator--' can be defaulted}}
};
struct PrefixNonMember {};
PrefixNonMember &operator++(PrefixNonMember &) = default; // expected-error {{only the postfix form of 'operator++' can be defaulted}}

// Other operators still cannot be defaulted.
struct Other {
  Other operator+(int) = default; // expected-error {{only special member functions, comparison operators, and postfix increment and decrement operators may be defaulted}}
};

// Dependent declarations are checked when instantiated.
template <typename T> struct DependentReturn {
  DependentReturn &operator++();
  T operator++(int) = default; // expected-error {{return type for defaulted postfix increment operator must be 'DependentReturn<int>', not 'int'}}
};
DependentReturn<int> dr1; // expected-note {{in instantiation of template class 'DependentReturn<int>' requested here}}

template <typename T> struct DependentObject {
  DependentObject &operator++();
  DependentObject operator++(this T &, int) = default; // expected-error {{invalid explicit object parameter type for defaulted postfix increment operator; found 'int &', expected 'DependentObject<int> &' or 'volatile DependentObject<int> &'}}
};
DependentObject<int> do1; // expected-note {{in instantiation of template class 'DependentObject<int>' requested here}}

template <typename T> struct DependentFriend {
  DependentFriend &operator++();
  friend DependentFriend operator++(DependentFriend &, int) = default; // OK
  friend int operator--(T &, int) = default;                           // expected-error {{return type for defaulted postfix decrement operator must be 'S', not 'int'}}
};
DependentFriend<S> df1; // expected-note {{in instantiation of template class 'DependentFriend<S>' requested here}}

// A deduced return type is never valid, so it is diagnosed even if C is not
// known yet.
template <typename T> struct DependentAuto {
  friend auto operator++(T &, int) = default; // expected-error {{defaulted postfix increment operator cannot have a deduced return type}}
  auto operator--(this T &, int) = default;   // expected-error {{defaulted postfix decrement operator cannot have a deduced return type}}
};

template <typename T> struct DependentDeducedReturn {
  DependentDeducedReturn &operator++();
  DependentDeducedReturn &operator--();
  auto operator++(int) = default;     // expected-error {{defaulted postfix increment operator cannot have a deduced return type}}
  Any auto operator--(int) = default; // expected-error {{defaulted postfix decrement operator cannot have a deduced return type}}
};
template <typename T> struct DependentDeducedReturn2 {
  DependentDeducedReturn2 &operator++();
  DependentDeducedReturn2 &operator--();
  decltype(auto) operator++(int) = default;                // expected-error {{defaulted postfix increment operator cannot have a deduced return type}}
  friend T *operator--(DependentDeducedReturn2 &, int) = default; // OK, checked when instantiated
};
template <typename T> struct DependentConstrainedFriend {
  DependentConstrainedFriend &operator++();
  friend Any auto operator++(DependentConstrainedFriend &, int) = default; // expected-error {{defaulted postfix increment operator cannot have a deduced return type}}
};

// A trailing return type that names C is not deduced.
template <typename T> struct DependentTrailing {
  DependentTrailing &operator++();
  auto operator++(int) -> T = default; // expected-error {{return type for defaulted postfix increment operator must be 'DependentTrailing<int>', not 'int'}}
};
DependentTrailing<int> dt1; // expected-note {{in instantiation of template class 'DependentTrailing<int>' requested here}}
template <typename T> struct DependentTrailingOK {
  DependentTrailingOK &operator++();
  auto operator++(int) -> DependentTrailingOK = default; // OK
};
DependentTrailingOK<int> dt2;

// A placeholder parameter makes the function a template even in a class
// template.
template <typename T> struct DependentAutoParam {
  DependentAutoParam &operator++();
  DependentAutoParam &operator--();
  DependentAutoParam operator++(this auto &, int) = default;  // expected-error {{postfix increment operator template cannot be defaulted}}
  friend DependentAutoParam operator--(auto &, int) = default; // expected-error {{postfix decrement operator template cannot be defaulted}}
  DependentAutoParam operator++(auto) = default;               // expected-error {{postfix increment operator template cannot be defaulted}}
};

// A dependent second parameter is checked when instantiated.
template <typename T> struct DependentSecond {
  DependentSecond &operator++();
  DependentSecond operator++(T) = default; // expected-error {{parameter of overloaded post-increment operator must have type 'int' (not 'long')}}
};
DependentSecond<int> ds1;  // OK
DependentSecond<long> ds2; // expected-note {{in instantiation of template class 'DependentSecond<long>' requested here}}

template <typename T> struct DependentRestrict {
  DependentRestrict &operator++();
  DependentRestrict operator++(int) __restrict__ = default;       // OK
  DependentRestrict operator--(int) const __restrict__ = default; // expected-error {{defaulted member postfix decrement operator cannot be const-qualified}}
};
DependentRestrict<int> dres1; // expected-note {{in instantiation of template class 'DependentRestrict<int>' requested here}}
