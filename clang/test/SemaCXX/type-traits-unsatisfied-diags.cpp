// RUN: %clang_cc1 -fsyntax-only -verify -Wno-vla-cxx-extension -Wno-c++26-extensions -std=c++20 %s

struct S : A {}; // expected-error{{expected class name}}

static_assert(__builtin_is_cpp_trivially_relocatable()); // expected-error {{expected a type}}
static_assert(__builtin_is_cpp_trivially_relocatable(0)); // expected-error {{expected a type}}
static_assert(__builtin_is_cpp_trivially_relocatable(S));
static_assert(__builtin_is_cpp_trivially_relocatable(A)); // expected-error{{unknown type name 'A'}}

static_assert(__builtin_is_cpp_trivially_relocatable(int, int)); // expected-error {{type trait requires 1 argument; have 2 arguments}}

static_assert(__builtin_is_cpp_trivially_relocatable(int&));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_cpp_trivially_relocatable(int &)'}} \
// expected-note@-1 {{'int &' is not trivially relocatable}} \
// expected-note@-1 {{because it is a reference type}}


static_assert(!__builtin_is_cpp_trivially_relocatable(int&));
static_assert(!!__builtin_is_cpp_trivially_relocatable(int&));
// expected-error@-1{{static assertion failed due to requirement '!!__builtin_is_cpp_trivially_relocatable(int &)'}}
static_assert(bool(__builtin_is_cpp_trivially_relocatable(int&)));
// expected-error@-1{{static assertion failed due to requirement 'bool(__builtin_is_cpp_trivially_relocatable(int &))'}}

static_assert(__builtin_is_cpp_trivially_relocatable(int&) && __builtin_is_cpp_trivially_relocatable(int&));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_cpp_trivially_relocatable(int &)'}} \
// expected-note@-1 {{'int &' is not trivially relocatable}} \
// expected-note@-1 {{because it is a reference type}}

namespace concepts {
template <typename T>
requires __builtin_is_cpp_trivially_relocatable(T) void f();  // #cand1

template <typename T>
concept C = __builtin_is_cpp_trivially_relocatable(T); // #concept2

template <C T> void g();  // #cand2

void test() {
    f<int&>();
    // expected-error@-1 {{no matching function for call to 'f'}} \
    // expected-note@#cand1 {{candidate template ignored: constraints not satisfied [with T = int &]}} \
    // expected-note@#cand1 {{because '__builtin_is_cpp_trivially_relocatable(int &)' evaluated to false}} \
    // expected-note@#cand1 {{'int &' is not trivially relocatable}} \
    // expected-note@#cand1 {{because it is a reference type}}

    g<int&>();
    // expected-error@-1 {{no matching function for call to 'g'}} \
    // expected-note@#cand2 {{candidate template ignored: constraints not satisfied [with T = int &]}} \
    // expected-note@#cand2 {{because 'int &' does not satisfy 'C'}} \
    // expected-note@#concept2 {{because '__builtin_is_cpp_trivially_relocatable(int &)' evaluated to false}} \
    // expected-note@#concept2 {{'int &' is not trivially relocatable}} \
    // expected-note@#concept2 {{because it is a reference type}}
}
}

namespace trivially_relocatable {

extern int vla_size;
static_assert(__builtin_is_cpp_trivially_relocatable(int[vla_size]));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_cpp_trivially_relocatable(int[vla_size])'}} \
// expected-note@-1 {{'int[vla_size]' is not trivially relocatable}} \
// expected-note@-1 {{because it is a variably-modified type}}

struct S; // expected-note {{forward declaration of 'trivially_relocatable::S'}}
static_assert(__builtin_is_cpp_trivially_relocatable(S));
// expected-error@-1 {{incomplete type 'S' used in type trait expression}}

struct B {
 virtual ~B();
};
struct S : virtual B { // #tr-S
    S();
    int & a;
    const int ci;
    B & b;
    B c;
    ~S();
};
static_assert(__builtin_is_cpp_trivially_relocatable(S));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_cpp_trivially_relocatable(trivially_relocatable::S)'}} \
// expected-note@-1 {{'S' is not trivially relocatable}} \
// expected-note@-1 {{because it has a virtual base 'B'}} \
// expected-note@-1 {{because it has a non-trivially-relocatable base 'B'}} \
// expected-note@-1 {{because it has a non-trivially-relocatable member 'c' of type 'B'}} \
// expected-note@-1 {{because it has a user-provided destructor}}
// expected-note@#tr-S {{'S' defined here}}

struct S2 { // #tr-S2
    S2(S2&&);
    S2& operator=(const S2&);
};
static_assert(__builtin_is_cpp_trivially_relocatable(S2));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_cpp_trivially_relocatable(trivially_relocatable::S2)'}} \
// expected-note@-1 {{'S2' is not trivially relocatable}} \
// expected-note@-1 {{because it has a user provided move constructor}} \
// expected-note@-1 {{because it has a user provided copy assignment operator}} \
// expected-note@#tr-S2 {{'S2' defined here}}


struct S3 { // #tr-S3
    ~S3() = delete;
};
static_assert(__builtin_is_cpp_trivially_relocatable(S3));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_cpp_trivially_relocatable(trivially_relocatable::S3)'}} \
// expected-note@-1 {{'S3' is not trivially relocatable}} \
// expected-note@-1 {{because it has a deleted destructor}} \
// expected-note@#tr-S3 {{'S3' defined here}}


union U { // #tr-U
    U(const U&);
    U(U&&);
    U& operator=(const U&);
    U& operator=(U&&);
};
static_assert(__builtin_is_cpp_trivially_relocatable(U));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_cpp_trivially_relocatable(trivially_relocatable::U)'}} \
// expected-note@-1 {{'U' is not trivially relocatable}} \
// expected-note@-1 {{because it is a union with a user-declared copy constructor}} \
// expected-note@-1 {{because it is a union with a user-declared copy assignment operator}} \
// expected-note@-1 {{because it is a union with a user-declared move constructor}} \
// expected-note@-1 {{because it is a union with a user-declared move assignment operator}}
// expected-note@#tr-U {{'U' defined here}}
struct S4 trivially_relocatable_if_eligible { // #tr-S4
    ~S4();
    B b;
};
static_assert(__builtin_is_cpp_trivially_relocatable(S4));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_cpp_trivially_relocatable(trivially_relocatable::S4)'}} \
// expected-note@-1 {{'S4' is not trivially relocatable}} \
// expected-note@-1 {{because it has a non-trivially-relocatable member 'b' of type 'B'}} \
// expected-note@#tr-S4 {{'S4' defined here}}

union U2 trivially_relocatable_if_eligible { // #tr-U2
    U2(const U2&);
    U2(U2&&);
    B b;
};
static_assert(__builtin_is_cpp_trivially_relocatable(U2));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_cpp_trivially_relocatable(trivially_relocatable::U2)'}} \
// expected-note@-1 {{'U2' is not trivially relocatable}} \
// expected-note@-1 {{because it has a deleted destructor}} \
// expected-note@-1 {{because it has a non-trivially-relocatable member 'b' of type 'B'}} \
// expected-note@#tr-U2 {{'U2' defined here}}

}
