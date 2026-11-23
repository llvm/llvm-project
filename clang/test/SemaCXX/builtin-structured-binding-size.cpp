// RUN: %clang_cc1 %s -triple=x86_64 -std=c++2c -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple=x86_64 -std=c++2c -fsyntax-only -verify -fexperimental-new-constant-interpreter


struct S0 {};
struct S1 {int a;};
struct S2 {int a; int b; static int c;};
struct S3 {double a; int b; int c;};
struct S4 {int a: 1; int b :2;};
struct S5 {int : 1; int b :2;};
struct S6 {union {int a;}; }; // #note-anon-union
struct S7 {int a[];};



struct SD : S1 {};
struct SE1 : S1 { int b;};

class  P1 {int a;}; // #note-private

union U1 {};
union U2 {int a;};

template <typename T>
concept is_destructurable = requires {
    { __builtin_structured_binding_size(T) };
};

static_assert(__builtin_structured_binding_size(S0) == 0);
static_assert(__is_same_as(decltype(__builtin_structured_binding_size(S0)), decltype(sizeof(void*))));

static_assert(__builtin_structured_binding_size(S1) == 0);
// expected-error@-1 {{static assertion failed due to requirement '__builtin_structured_binding_size(S1) == 0'}} \
// expected-note@-1 {{expression evaluates to '1 == 0'}}
static_assert(__builtin_structured_binding_size(S1) == 1);
static_assert(__builtin_structured_binding_size(S2) == 2);
static_assert(__builtin_structured_binding_size(S3) == 3);
static_assert(__builtin_structured_binding_size(S4) == 2);
static_assert(__builtin_structured_binding_size(S5) == 2);
// expected-error@-1 {{static assertion failed due to requirement '__builtin_structured_binding_size(S5) == 2'}} \
// expected-note@-1 {{expression evaluates to '1 == 2'}}
static_assert(__builtin_structured_binding_size(S6) == 2);
// expected-error@-1 {{cannot bind class type 'S6' because it has an anonymous union member}} \
// expected-error@-1 {{type 'S6' cannot be bound}} \
// expected-error@-1 {{static assertion expression is not an integral constant expression}} \
// expected-note@#note-anon-union {{declared here}}
static_assert(__builtin_structured_binding_size(S7) == 1);


static_assert(__builtin_structured_binding_size(SD) == 1);
static_assert(__builtin_structured_binding_size(SE1) == 1);
// expected-error@-1 {{cannot bind class type 'SE1': both it and its base class 'S1' have non-static data members}} \
// expected-error@-1 {{type 'SE1' cannot be bound}} \
// expected-error@-1 {{static assertion expression is not an integral constant expression}}

static_assert(__builtin_structured_binding_size(U1) == 0);
// expected-error@-1 {{type 'U1' cannot be bound}} \
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
static_assert(__builtin_structured_binding_size(U2) == 0);
// expected-error@-1 {{type 'U2' cannot be bound}} \
// expected-error@-1 {{static assertion expression is not an integral constant expression}}



static_assert(__builtin_structured_binding_size(int[0]) == 0);
static_assert(__builtin_structured_binding_size(int[1]) == 1);
static_assert(__builtin_structured_binding_size(int[42]) == 42);

using vec2 = int __attribute__((__vector_size__(2 * sizeof(int))));
using vec3 = int __attribute__((__vector_size__(3 * sizeof(int))));
static_assert(__builtin_structured_binding_size(vec2) == 2);
static_assert(__builtin_structured_binding_size(vec3) == 3);
static_assert(__builtin_structured_binding_size(decltype(__builtin_complex(0., 0.))) == 2);


int VLASize; // expected-note {{declared here}}
static_assert(__builtin_structured_binding_size(int[VLASize]) == 42);
// expected-error@-1 {{type 'int[VLASize]' cannot be bound}} \
// expected-warning@-1 {{variable length arrays in C++ are a Clang extension}} \
// expected-note@-1 {{read of non-const variable 'VLASize' is not allowed in a constant expression}} \
// expected-error@-1 {{static assertion expression is not an integral constant expression}}


struct Incomplete; // expected-note {{forward declaration of 'Incomplete'}}
static_assert(__builtin_structured_binding_size(Incomplete) == 1);
// expected-error@-1 {{incomplete type 'Incomplete' where a complete type is required}} \
// expected-error@-1 {{type 'Incomplete' cannot be bound}} \
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
static_assert(__builtin_structured_binding_size(Incomplete[]) == 1);
// expected-error@-1 {{type 'Incomplete[]' cannot be bound}} \
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
static_assert(__builtin_structured_binding_size(Incomplete[0]) == 0);
static_assert(__builtin_structured_binding_size(Incomplete[1]) == 1);
static_assert(__builtin_structured_binding_size(Incomplete[42]) == 42);


static_assert(__builtin_structured_binding_size(P1) == 0);
// expected-error@-1 {{static assertion failed due to requirement '__builtin_structured_binding_size(P1) == 0'}} \
// expected-note@-1 {{expression evaluates to '1 == 0'}} \
// expected-error@-1 {{cannot bind private member 'a' of 'P1}} \
// expected-note@#note-private {{implicitly declared private here}}


void func(int array[14], int x = __builtin_structured_binding_size(decltype(array)));
//expected-error@-1 {{type 'decltype(array)' (aka 'int *') cannot be bound}}

struct SM {
    static int array[14];
    static_assert(__builtin_structured_binding_size(decltype(array)) == 14);
};

template <typename Ty, int N = __builtin_structured_binding_size(Ty)> // #tpl-1
struct T {
    static constexpr int value = N;
};

T<int> t1;
// expected-error@#tpl-1 {{type 'int' cannot be bound}} \
// expected-error@#tpl-1 {{non-type template argument is not a constant expression}} \
// expected-note@-1 {{in instantiation of default argument for 'T<int>' required here}} \
// expected-note@-1 {{while checking a default template argument used here}} \

static_assert(T<S3>::value == 3);

static_assert(is_destructurable<S0>);
static_assert(is_destructurable<const S0>);
static_assert(is_destructurable<volatile S0>);
static_assert(!is_destructurable<S0&>);
static_assert(is_destructurable<S1>);
static_assert(!is_destructurable<S1&>);
static_assert(!is_destructurable<SE1>);
static_assert(!is_destructurable<int>);
static_assert(!is_destructurable<int[]>);
static_assert(is_destructurable<int[1]>);
static_assert(!is_destructurable<P1>);

template <typename T>
constexpr int f() {return 0;};
template <typename T>
requires is_destructurable<T>
constexpr int f() {return 1;};

static_assert(f<int>() == 0);
static_assert(f<S0>()  == 1);

struct T0;
struct T1;
struct T42;
struct TSizeError;

namespace std {

template <typename>
struct tuple_size;

template <>
struct tuple_size<T0> {
    static constexpr int value = 0;
};

template <>
struct tuple_size<T1> {
    static constexpr int value = 1;
};

template <>
struct tuple_size<T42> {
    static constexpr int value = 42;
};

template <>
struct tuple_size<TSizeError> {
    static constexpr void* value = nullptr;
};

static_assert(__builtin_structured_binding_size(T0) == 0);

static_assert(is_destructurable<const T0>);
static_assert(is_destructurable<volatile T0>);
static_assert(!is_destructurable<T0&>);


static_assert(__builtin_structured_binding_size(T1) == 1);
static_assert(__builtin_structured_binding_size(T42) == 42);
static_assert(__builtin_structured_binding_size(TSizeError) == 42);
// expected-error@-1 {{cannot bind this type; 'std::tuple_size<TSizeError>::value' is not a valid integral constant expression}} \
// expected-error@-1 {{type 'TSizeError' cannot be bound}} \
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
static_assert(!is_destructurable<TSizeError>);
}


struct S {
  int x;
  int y;
  static_assert(__builtin_structured_binding_size(S) == 2);
  //expected-error@-1 {{incomplete type 'S' where a complete type is required}} \
  // expected-error@-1 {{type 'S' cannot be bound}} \
  // expected-error@-1 {{static assertion expression is not an integral constant expression}} \
  // expected-note@-4 {{definition of 'S' is not complete until the closing '}'}}
};

// Check we can implement std::exec::tag_of_t
template <typename T>
struct type_identity {
    using type = T;
};
template<typename T> T &&declval();

template <typename T>
requires (__builtin_structured_binding_size(T) >=2)
consteval auto tag_of_impl(T& t) {
    auto && [tag, ..._] = t;
    return type_identity<decltype(auto(tag))>{};
}

template <typename T>
requires (__builtin_structured_binding_size(T) >=2) // #tag-of-constr
using tag_of_t = decltype(tag_of_impl(declval<T&>()))::type;

static_assert(__is_same_as(tag_of_t<S2>, int));
static_assert(__is_same_as(tag_of_t<S3>, double));


static_assert(__is_same_as(tag_of_t<S1>, int));
// expected-error@-1 {{constraints not satisfied for alias template 'tag_of_t' [with T = S1]}} \
// expected-note@#tag-of-constr {{because '__builtin_structured_binding_size(S1) >= 2' (1 >= 2) evaluated to false}}

static_assert(__is_same_as(tag_of_t<int>, int)); // error
// expected-error@-1 {{constraints not satisfied for alias template 'tag_of_t' [with T = int]}}
// expected-note@#tag-of-constr {{because substituted constraint expression is ill-formed: type 'int' cannot be bound}}

struct MinusOne;
template <> struct ::std::tuple_size<MinusOne> {
  static constexpr int value = -1;
};
int minus_one = __builtin_structured_binding_size(MinusOne);
// expected-error@-1 {{cannot bind this type; 'std::tuple_size<MinusOne>::value' is not a valid size: -1}}
// expected-error@-2 {{type 'MinusOne' cannot be bound}}

struct UintMax;
template <> struct ::std::tuple_size<UintMax> {
  static constexpr unsigned value = -1;
};
int uint_max = __builtin_structured_binding_size(UintMax);
// expected-error@-1 {{cannot bind this type; 'std::tuple_size<UintMax>::value' is not a valid size: 4294967295}}
// expected-error@-2 {{type 'UintMax' cannot be bound}}
