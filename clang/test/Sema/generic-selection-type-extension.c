// RUN: %clang_cc1 -std=c2x -fsyntax-only -verify -Wno-unused %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unused -x c++ -std=c++17 %s

// Test that the semantic behavior of the extension allowing the user to pass a
// type as the first argument to _Generic.

// Test that we match on basic types.
static_assert(_Generic(int, int : 1, default : 0) == 1);
static_assert(_Generic(_BitInt(12), int : 1, _BitInt(10) : 2, _BitInt(12) : 3) == 3);

// Test that we correctly fall back to the default association appropriately.
static_assert(_Generic(int, long : 1, default : 0) == 0);

// Ensure we correctly match constant arrays by their extent.
static_assert(_Generic(int[12], int[0] : 0, int * : 0, int[12] : 1, default : 0) == 1);

// Ensure we correctly match function types by their signature.
static_assert(_Generic(int(int), void(void) : 0, int(void) : 0, void(int) : 0, int(int) : 1, default : 0) == 1);

// Test that we still diagnose when no associations match and that the
// diagnostic includes qualifiers.
static_assert(_Generic(const int, long : 1)); // expected-error {{controlling expression type 'const int' not compatible with any generic association type}}

// Test that qualifiers work as expected and do not issue a diagnostic when
// using the type form.
static_assert(_Generic(const int, int : 0, const int : 1) == 1);
static_assert(_Generic(int volatile _Atomic const, int : 0, const int : 0, volatile int : 0, _Atomic int : 0, _Atomic const volatile int : 1) == 1);

// Test that inferred qualifiers also work as expected.
const int ci = 0;
static_assert(_Generic(__typeof__(ci), int : 0, const int : 1) == 1);
// And that the expression form still complains about qualified associations
// and matches the correct association.
static_assert(_Generic(ci, int : 1, const int : 0) == 1); // expected-warning {{due to lvalue conversion of the controlling expression, association of type 'const int' will never be selected because it is qualified}}

// The type operand form of _Generic allows incomplete and non-object types,
// but the expression operand form still rejects them.
static_assert(_Generic(struct incomplete, struct incomplete : 1, default : 0) == 1);
static_assert(_Generic(struct another_incomplete, struct incomplete : 1, default : 0) == 0);
static_assert(_Generic(1, struct also_incomplete : 1, default : 0) == 0); // expected-error {{type 'struct also_incomplete' in generic association incomplete}}

void foo(int);
static_assert(_Generic(__typeof__(foo), void(int) : 1, default : 0) == 1);
static_assert(_Generic(foo, void(int) : 1, default : 0) == 0); // expected-error {{type 'void (int)' in generic association not an object type}}

// Ensure we still get a diagnostic for duplicated associations for the type
// form, even when using qualified type, and that the diagnostic includes
// qualifiers.
static_assert(_Generic(const int,
                         const int : 1, // expected-note {{compatible type 'const int' specified here}}
                         int : 2,
                         const int : 3  // expected-error {{type 'const int' in generic association compatible with previously specified type 'const int'}}
                      ) == 1);

// Verify that we are matching using the canonical type of the type operand...
typedef int Int;
typedef const Int CInt;
typedef CInt OtherCInt;
static_assert(_Generic(volatile CInt, const volatile int : 1, default : 0) == 1);
static_assert(_Generic(const int, CInt : 1, default : 0) == 1);

// ...and that duplicate associations are doing so as well.
static_assert(_Generic(const int,
                         CInt : 1,     // expected-note {{compatible type 'CInt' (aka 'const int') specified here}}
                         const volatile int : 2,
                         OtherCInt : 3 // expected-error {{type 'OtherCInt' (aka 'const int') in generic association compatible with previously specified type 'CInt' (aka 'const int')}}
                      ) == 1);

// Also test that duplicate array or function types are caught.
static_assert(_Generic(const int,
                         int[12] : 0,  // expected-note {{compatible type 'int[12]' specified here}}
                         int[12] : 0,  // expected-error {{type 'int[12]' in generic association compatible with previously specified type 'int[12]'}}
                         int(int) : 0, // expected-note {{compatible type 'int (int)' specified here}}
                         int(int) : 0, // expected-error {{type 'int (int)' in generic association compatible with previously specified type 'int (int)'}}
                         default : 1
                      ) == 1);


// Tests that only make sense for C++:
#ifdef __cplusplus
// Ensure that _Generic works within a template argument list.
template <typename Ty, int N = _Generic(Ty, int : 0, default : 1)>
constexpr Ty bar() { return N; }

static_assert(bar<int>() == 0);
static_assert(bar<float>() == 1);

// Or that it can be used as a non-type template argument.
static_assert(bar<int, _Generic(int, int : 1, default : 0)>() == 1);

// Ensure that a dependent type works as expected.
template <typename Ty>
struct Dependent {
  // If we checked the type early, this would fail to compile without any
  // instantiation. Instead, it only fails with the bad instantiation.
  static_assert(_Generic(Ty, int : 1)); // expected-error {{controlling expression type 'double' not compatible with any generic association type}} \
                                           expected-note@#BadInstantiation {{in instantiation of template class 'Dependent<double>' requested here}}
};

template struct Dependent<int>; // Good instantiation
template struct Dependent<double>; // #BadInstantiation

// Another template instantiation test, this time for a variable template with
// a type-dependent initializer.
template <typename Ty>
constexpr auto Val = _Generic(Ty, Ty : Ty{});

static_assert(Val<int> == 0);
static_assert(__is_same(decltype(Val<Dependent<int>>), const Dependent<int>));

// Ensure that pack types also work as expected.
template <unsigned Arg, unsigned... Args> struct Or {
  enum { result = Arg | Or<Args...>::result };
};

template <unsigned Arg> struct Or<Arg> {
  enum { result = Arg };
};

template <class... Args> struct TypeMask {
  enum {
   result = Or<_Generic(Args, int: 1, long: 2, short: 4, float: 8)...>::result
  };
};

static_assert(TypeMask<int, long, short>::result == 7, "fail");
static_assert(TypeMask<float, short>::result == 12, "fail");
static_assert(TypeMask<int, float, float>::result == 9, "fail");

template <typename... T>
void f() {
  // Because _Generic only accepts a single type argument, it does not make
  // sense for it to accept a pack, so a pack is rejected while parsing.
  _Generic(T..., int : 1); // expected-error {{expected ','}}
}
#endif // __cplusplus
