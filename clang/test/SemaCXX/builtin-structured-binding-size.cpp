// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -verify
// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -verify -fexperimental-new-constant-interpreter


struct S0 {};
struct S1 {int a;};
struct S2 {int a; int b;};
struct S3 {double a; int b; int c;};



struct SD : S1 {};
struct SE1 : S1 { int b;};

class  P1 {int a;}; // #note-private


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
static_assert(__builtin_structured_binding_size(SD) == 1);
static_assert(__builtin_structured_binding_size(SE1) == 1);
// expected-error@-1 {{cannot decompose class type 'SE1': both it and its base class 'S1' have non-static data members}} \
// expected-error@-1 {{type 'SE1' is not destructurable}} \
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
// expected-error@-1 {{type 'int[VLASize]' is not destructurable}} \
// expected-warning@-1 {{variable length arrays in C++ are a Clang extension}} \
// expected-note@-1 {{read of non-const variable 'VLASize' is not allowed in a constant expression}} \
// expected-error@-1 {{static assertion expression is not an integral constant expression}}


struct Incomplete; // expected-note {{forward declaration of 'Incomplete'}}
static_assert(__builtin_structured_binding_size(Incomplete) == 1);
// expected-error@-1 {{incomplete type 'Incomplete' where a complete type is required}} \
// expected-error@-1 {{type 'Incomplete' is not destructurable}} \
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
static_assert(__builtin_structured_binding_size(Incomplete[]) == 1);
// expected-error@-1 {{type 'Incomplete[]' is not destructurable}} \
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
static_assert(__builtin_structured_binding_size(Incomplete[0]) == 0);
static_assert(__builtin_structured_binding_size(Incomplete[1]) == 1);
static_assert(__builtin_structured_binding_size(Incomplete[42]) == 42);


static_assert(__builtin_structured_binding_size(P1) == 0);
// expected-error@-1 {{static assertion failed due to requirement '__builtin_structured_binding_size(P1) == 0'}} \
// expected-note@-1 {{expression evaluates to '1 == 0'}} \
// expected-error@-1 {{cannot decompose private member 'a' of 'P1}} \
// expected-note@#note-private {{implicitly declared private here}}


static_assert(is_destructurable<S0>);
static_assert(is_destructurable<S1>);
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
static_assert(__builtin_structured_binding_size(T1) == 1);
static_assert(__builtin_structured_binding_size(T42) == 42);
static_assert(__builtin_structured_binding_size(TSizeError) == 42);
// expected-error@-1 {{cannot decompose this type; 'std::tuple_size<TSizeError>::value' is not a valid integral constant expression}} \
// expected-error@-1 {{type 'TSizeError' is not destructurable}} \
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
static_assert(!is_destructurable<TSizeError>);
}


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
// expected-note@#tag-of-constr {{because substituted constraint expression is ill-formed: type 'int' is not destructurable}}
