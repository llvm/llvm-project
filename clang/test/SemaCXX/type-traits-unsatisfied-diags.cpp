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

namespace replaceable {

extern int vla_size;
static_assert(__builtin_is_replaceable(int[vla_size]));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_replaceable(int[vla_size])'}} \
// expected-note@-1 {{'int[vla_size]' is not replaceable}} \
// expected-note@-1 {{because it is a variably-modified type}}

struct S; // expected-note {{forward declaration of 'replaceable::S'}}
static_assert(__builtin_is_replaceable(S));
// expected-error@-1 {{incomplete type 'S' used in type trait expression}}

static_assert(__builtin_is_replaceable(const volatile int));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_replaceable(const volatile int)}} \
// expected-note@-1 {{'const volatile int' is not replaceable}} \
// expected-note@-1 {{because it is const}} \
// expected-note@-1 {{because it is volatile}}


static_assert(__builtin_is_replaceable(void()));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_replaceable(void ())}} \
// expected-note@-1 {{'void ()' is not replaceable}} \
// expected-note@-1 {{because it is not a scalar or class type}}

struct B {
 virtual ~B();
};
struct S : virtual B { // #replaceable-S
    S();
    int & a;
    const int ci;
    B & b;
    B c;
    ~S();
};
static_assert(__builtin_is_replaceable(S));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_replaceable(replaceable::S)'}} \
// expected-note@-1 {{'S' is not replaceable}} \
// expected-note@-1 {{because it has a non-replaceable base 'B'}} \
// expected-note@-1 {{because it has a non-replaceable member 'a' of type 'int &'}} \
// expected-note@-1 {{because it has a non-replaceable member 'ci' of type 'const int'}} \
// expected-note@-1 {{because it has a non-replaceable member 'b' of type 'B &'}} \
// expected-note@-1 {{because it has a non-replaceable member 'c' of type 'B'}} \
// expected-note@-1 {{because it has a user-provided destructor}} \
// expected-note@-1 {{because it has a deleted copy assignment operator}}
// expected-note@#replaceable-S {{'S' defined here}}

struct S2 { // #replaceable-S2
    S2(S2&&);
    S2& operator=(const S2&);
};
static_assert(__builtin_is_replaceable(S2));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_replaceable(replaceable::S2)'}} \
// expected-note@-1 {{'S2' is not replaceable}} \
// expected-note@-1 {{because it has a user provided move constructor}} \
// expected-note@-1 {{because it has a user provided copy assignment operator}} \
// expected-note@#replaceable-S2 {{'S2' defined here}}


struct S3 { // #replaceable-S3
    ~S3() = delete;
};
static_assert(__builtin_is_replaceable(S3));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_replaceable(replaceable::S3)'}} \
// expected-note@-1 {{'S3' is not replaceable}} \
// expected-note@-1 {{because it has a deleted destructor}} \
// expected-note@#replaceable-S3 {{'S3' defined here}}


union U { // #replaceable-U
    U(const U&);
    U(U&&);
    U& operator=(const U&);
    U& operator=(U&&);
};
static_assert(__builtin_is_replaceable(U));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_replaceable(replaceable::U)'}} \
// expected-note@-1 {{'U' is not replaceable}} \
// expected-note@-1 {{because it is a union with a user-declared copy constructor}} \
// expected-note@-1 {{because it is a union with a user-declared copy assignment operator}} \
// expected-note@-1 {{because it is a union with a user-declared move constructor}} \
// expected-note@-1 {{because it is a union with a user-declared move assignment operator}}
// expected-note@#replaceable-U {{'U' defined here}}
struct S4 replaceable_if_eligible { // #replaceable-S4
    ~S4();
    B b;
};
static_assert(__builtin_is_replaceable(S4));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_replaceable(replaceable::S4)'}} \
// expected-note@-1 {{'S4' is not replaceable}} \
// expected-note@-1 {{because it has a non-replaceable member 'b' of type 'B'}} \
// expected-note@#replaceable-S4 {{'S4' defined here}}

union U2 replaceable_if_eligible { // #replaceable-U2
    U2(const U2&);
    U2(U2&&);
    B b;
};
static_assert(__builtin_is_replaceable(U2));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_replaceable(replaceable::U2)'}} \
// expected-note@-1 {{'U2' is not replaceable}} \
// expected-note@-1 {{because it has a deleted destructor}} \
// expected-note@-1 {{because it has a non-replaceable member 'b' of type 'B'}} \
// expected-note@-1 {{because it has a deleted copy assignment operator}} \
// expected-note@#replaceable-U2 {{'U2' defined here}}

struct UD1 {  // #replaceable-UD1
    UD1(const UD1&) = delete;
    UD1 & operator=(const UD1&) = delete;

};
static_assert(__builtin_is_replaceable(UD1));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_replaceable(replaceable::UD1)'}} \
// expected-note@-1 {{'UD1' is not replaceable}} \
// expected-note@-1 {{because it has a deleted copy constructor}} \
// expected-note@-1 {{because it has a deleted copy assignment operator}} \
// expected-note@#replaceable-UD1 {{'UD1' defined here}}


struct UD2 {  // #replaceable-UD2
    UD2(UD2&&) = delete;
    UD2 & operator=(UD2&&) = delete;
};
static_assert(__builtin_is_replaceable(UD2));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_replaceable(replaceable::UD2)'}} \
// expected-note@-1 {{'UD2' is not replaceable}} \
// expected-note@-1 {{because it has a deleted move constructor}} \
// expected-note@-1 {{because it has a deleted move assignment operator}} \
// expected-note@#replaceable-UD2 {{'UD2' defined here}}

}


namespace GH143325 {
struct Foo  { // expected-note {{previous definition is here}}
  Foo(const Foo&);
  ~Foo();
};

struct Foo { // expected-error {{redefinition of 'Foo'}}
  Foo();
  int;
};
struct Wrapper { // #GH143325-Wrapper
  union {
    Foo p;
  } u;
};

static_assert(__builtin_is_cpp_trivially_relocatable(Wrapper));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_cpp_trivially_relocatable(GH143325::Wrapper)'}} \
// expected-note@-1 {{'Wrapper' is not trivially relocatable}} \
// expected-note@-1 {{because it has a non-trivially-relocatable member 'u' of type 'union}} \
// expected-note@-1 {{because it has a deleted destructor}}
// expected-note@#GH143325-Wrapper {{'Wrapper' defined here}}

struct Polymorphic  {
  virtual ~Polymorphic();
};

struct UnionOfPolymorphic { // #GH143325-UnionOfPolymorphic
  union {
    Polymorphic p;
    int i;
  } u;
};

static_assert(__builtin_is_cpp_trivially_relocatable(UnionOfPolymorphic));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_cpp_trivially_relocatable(GH143325::UnionOfPolymorphic)'}} \
// expected-note@-1 {{'UnionOfPolymorphic' is not trivially relocatable}} \
// expected-note@-1 {{because it has a non-trivially-relocatable member 'u' of type 'union}} \
// expected-note@-1 {{because it has a deleted destructor}} \
// expected-note@#GH143325-UnionOfPolymorphic {{'UnionOfPolymorphic' defined here}}

}

struct GH143599 {  // expected-note 2 {{'GH143599' defined here}}
    ~GH143599 ();
     GH143599(const GH143599&);
     GH143599& operator=(const GH143599&);
};
GH143599::~GH143599 () = default;
GH143599::GH143599 (const GH143599&) = default;
GH143599& GH143599::operator=(const GH143599&) = default;

static_assert (__builtin_is_cpp_trivially_relocatable(GH143599));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_cpp_trivially_relocatable(GH143599)'}} \
// expected-note@-1 {{'GH143599' is not trivially relocatable}} \
// expected-note@-1 {{because it has a user provided copy constructor}} \
// expected-note@-1 {{because it has a user provided copy assignment operator}} \
// expected-note@-1 {{because it has a user-provided destructor}}

static_assert (__builtin_is_replaceable(GH143599));
// expected-error@-1 {{static assertion failed due to requirement '__builtin_is_replaceable(GH143599)'}} \
// expected-note@-1 {{'GH143599' is not replaceable}} \
// expected-note@-1 {{because it has a user provided copy constructor}} \
// expected-note@-1 {{because it has a user provided copy assignment operator}} \
// expected-note@-1 {{because it has a user-provided destructor}}

namespace trivially_copyable {
struct B {
 virtual ~B();
};
struct S : virtual B { // #tc-S
    S();
    int & a;
    const int ci;
    B & b;
    B c;
    ~S();
};
static_assert(__is_trivially_copyable(S));
// expected-error@-1 {{static assertion failed due to requirement '__is_trivially_copyable(trivially_copyable::S)'}} \
// expected-note@-1 {{'S' is not trivially copyable}} \
// expected-note@-1 {{because it has a virtual base 'B'}} \
// expected-note@-1 {{because it has a non-trivially-copyable base 'B'}} \
// expected-note@-1 {{because it has a non-trivially-copyable member 'c' of type 'B'}} \
// expected-note@-1 {{because it has a non-trivially-copyable member 'b' of type 'B &'}} \
// expected-note@-1 {{because it has a non-trivially-copyable member 'a' of type 'int &'}} \
// expected-note@-1 {{because it has a user-provided destructor}}
// expected-note@#tc-S {{'S' defined here}}

struct S2 { // #tc-S2
    S2(S2&&);
    S2& operator=(const S2&);
};
static_assert(__is_trivially_copyable(S2));
// expected-error@-1 {{static assertion failed due to requirement '__is_trivially_copyable(trivially_copyable::S2)'}} \
// expected-note@-1 {{'S2' is not trivially copyable}} \
// expected-note@-1 {{because it has a user provided move constructor}} \
// expected-note@-1 {{because it has a user provided copy assignment operator}} \
// expected-note@#tc-S2 {{'S2' defined here}}

struct S3 {
    ~S3() = delete;
};
static_assert(__is_trivially_copyable(S3));

struct S4 { // #tc-S4
    ~S4();
    B b;
};
static_assert(__is_trivially_copyable(S4));
// expected-error@-1 {{static assertion failed due to requirement '__is_trivially_copyable(trivially_copyable::S4)'}} \
// expected-note@-1 {{'S4' is not trivially copyable}} \
// expected-note@-1 {{because it has a non-trivially-copyable member 'b' of type 'B'}} \
// expected-note@-1 {{because it has a user-provided destructor}} \
// expected-note@#tc-S4 {{'S4' defined here}}

struct B1 {
    int & a;
};

struct B2 {
    int & a;
};

struct S5 : virtual B1, virtual B2 { // #tc-S5
};
static_assert(__is_trivially_copyable(S5));
// expected-error@-1 {{static assertion failed due to requirement '__is_trivially_copyable(trivially_copyable::S5)'}} \
// expected-note@-1 {{'S5' is not trivially copyable}} \
// expected-note@-1 {{because it has a virtual base 'B1'}} \
// expected-note@-1 {{because it has a virtual base 'B2'}} \
// expected-note@#tc-S5 {{'S5' defined here}}

struct B3 {
    ~B3();
};

struct B4 {
    ~B4();
};

struct S6 : B3, B4 { // #tc-S6
};
static_assert(__is_trivially_copyable(S6));
// expected-error@-1 {{static assertion failed due to requirement '__is_trivially_copyable(trivially_copyable::S6)'}} \
// expected-note@-1 {{because it has a non-trivially-copyable base 'B3'}} \
// expected-note@-1 {{because it has a non-trivially-copyable base 'B4'}} \
// expected-note@-1 {{because it has a user-provided destructor}} \
// expected-note@-1 {{'S6' is not trivially copyable}} \
// expected-note@#tc-S6 {{'S6' defined here}}

struct S7 { // #tc-S7
    S7(const S7&);
};
static_assert(__is_trivially_copyable(S7));
// expected-error@-1 {{static assertion failed due to requirement '__is_trivially_copyable(trivially_copyable::S7)'}} \
// expected-note@-1 {{because it has a user provided copy constructor}} \
// expected-note@-1 {{'S7' is not trivially copyable}} \
// expected-note@#tc-S7 {{'S7' defined here}}

struct S8 { // #tc-S8
    S8(S8&&);
};
static_assert(__is_trivially_copyable(S8));
// expected-error@-1 {{static assertion failed due to requirement '__is_trivially_copyable(trivially_copyable::S8)'}} \
// expected-note@-1 {{because it has a user provided move constructor}} \
// expected-note@-1 {{'S8' is not trivially copyable}} \
// expected-note@#tc-S8 {{'S8' defined here}}

struct S9 { // #tc-S9
    S9& operator=(const S9&);
};
static_assert(__is_trivially_copyable(S9));
// expected-error@-1 {{static assertion failed due to requirement '__is_trivially_copyable(trivially_copyable::S9)'}} \
// expected-note@-1 {{because it has a user provided copy assignment operator}} \
// expected-note@-1 {{'S9' is not trivially copyable}} \
// expected-note@#tc-S9 {{'S9' defined here}}

struct S10 { // #tc-S10
    S10& operator=(S10&&);
};
static_assert(__is_trivially_copyable(S10));
// expected-error@-1 {{static assertion failed due to requirement '__is_trivially_copyable(trivially_copyable::S10)'}} \
// expected-note@-1 {{because it has a user provided move assignment operator}} \
// expected-note@-1 {{'S10' is not trivially copyable}} \
// expected-note@#tc-S10 {{'S10' defined here}}

struct B5 : B4 {
};

struct B6 : B5  {
};

struct S11 : B6 { // #tc-S11
};
static_assert(__is_trivially_copyable(S11));
// expected-error@-1 {{static assertion failed due to requirement '__is_trivially_copyable(trivially_copyable::S11)'}} \
// expected-note@-1 {{because it has a non-trivially-copyable base 'B6'}} \
// expected-note@-1 {{'S11' is not trivially copyable}} \
// expected-note@#tc-S11 {{'S11' defined here}}

struct S12 : B6 { // #tc-S12
    ~S12() = delete;
};
static_assert(__is_trivially_copyable(S12));
// expected-error@-1 {{static assertion failed due to requirement '__is_trivially_copyable(trivially_copyable::S12)'}} \
// expected-note@-1 {{because it has a non-trivially-copyable base 'B6'}} \
// expected-note@-1 {{because it has a deleted destructor}} \
// expected-note@-1 {{'S12' is not trivially copyable}} \
// expected-note@#tc-S12 {{'S12' defined here}}
}

namespace constructible {

struct S1 {  // #c-S1
    S1(int); // #cc-S1
};
static_assert(__is_constructible(S1, char*));
// expected-error@-1 {{static assertion failed due to requirement '__is_constructible(constructible::S1, char *)'}} \
// expected-error@-1 {{no matching constructor for initialization of 'S1'}} \
// expected-note@#c-S1 {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'char *' to 'const S1' for 1st argument}} \
// expected-note@#c-S1 {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'char *' to 'S1' for 1st argument}} \
// expected-note@#cc-S1 {{candidate constructor not viable: no known conversion from 'char *' to 'int' for 1st argument; dereference the argument with *}} \
// expected-note@#c-S1 {{'S1' defined here}}

struct S2 { // #c-S2
    S2(int, float, double); // #cc-S2
};
static_assert(__is_constructible(S2, float));
// expected-error@-1 {{static assertion failed due to requirement '__is_constructible(constructible::S2, float)'}} \
// expected-note@#c-S2 {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'float' to 'const S2' for 1st argument}} \
// expected-note@#c-S2 {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'float' to 'S2' for 1st argument}} \
// expected-error@-1 {{no matching constructor for initialization of 'S2'}} \
// expected-note@#cc-S2 {{candidate constructor not viable: requires 3 arguments, but 1 was provided}} \
// expected-note@#c-S2 {{'S2' defined here}}

static_assert(__is_constructible(S2, float, void));
// expected-error@-1 {{static assertion failed due to requirement '__is_constructible(constructible::S2, float, void)'}} \
// expected-note@#c-S2 {{candidate constructor (the implicit move constructor) not viable: requires 1 argument, but 2 were provided}} \
// expected-note@#c-S2 {{candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 2 were provided}} \
// expected-note@-1{{because it is a cv void type}} \
// expected-error@-1 {{no matching constructor for initialization of 'S2'}} \
// expected-note@#cc-S2 {{candidate constructor not viable: requires 3 arguments, but 2 were provided}} \
// expected-note@#c-S2 {{'S2' defined here}}

static_assert(__is_constructible(int[]));
// expected-error@-1 {{static assertion failed due to requirement '__is_constructible(int[])'}} \
// expected-note@-1 {{because it is an incomplete array type}}

static_assert(__is_constructible(void));
// expected-error@-1 {{static assertion failed due to requirement '__is_constructible(void)'}} \
// expected-note@-1 {{because it is a cv void type}}

static_assert(__is_constructible(void, void));
// expected-error@-1 {{static assertion failed due to requirement '__is_constructible(void, void)'}} \
// expected-note@-1 {{because it is a cv void type}}

static_assert(__is_constructible(const void));
// expected-error@-1 {{static assertion failed due to requirement '__is_constructible(const void)'}} \
// expected-note@-1 {{because it is a cv void type}}

static_assert(__is_constructible(volatile void));
// expected-error@-1 {{static assertion failed due to requirement '__is_constructible(volatile void)'}} \
// expected-note@-1 {{because it is a cv void type}}

static_assert(__is_constructible(int ()));
// expected-error@-1 {{static assertion failed due to requirement '__is_constructible(int ())'}} \
// expected-note@-1 {{because it is a function type}}

static_assert(__is_constructible(void (int, float)));
// expected-error@-1 {{static assertion failed due to requirement '__is_constructible(void (int, float))'}} \
// expected-note@-1 {{because it is a function type}}
}

namespace assignable {
struct S1;
static_assert(__is_assignable(S1&, const S1&));
// expected-error@-1 {{static assertion failed due to requirement '__is_assignable(assignable::S1 &, const assignable::S1 &)'}} \
// expected-error@-1 {{no viable overloaded '='}} \
// expected-note@-1 {{type 'S1' is incomplete}}

static_assert(__is_assignable(void, int));
// expected-error@-1 {{static assertion failed due to requirement '__is_assignable(void, int)'}} \
// expected-error@-1 {{expression is not assignable}}

static_assert(__is_assignable(int, int));
// expected-error@-1 {{static assertion failed due to requirement '__is_assignable(int, int)'}} \
// expected-error@-1 {{expression is not assignable}}

static_assert(__is_assignable(int*, int));
// expected-error@-1 {{static assertion failed due to requirement '__is_assignable(int *, int)'}} \
// expected-error@-1 {{expression is not assignable}}

static_assert(__is_assignable(int[], int));
// expected-error@-1 {{static assertion failed due to requirement '__is_assignable(int[], int)'}} \
// expected-error@-1 {{expression is not assignable}}

static_assert(__is_assignable(int&, void));
// expected-error@-1 {{static assertion failed due to requirement '__is_assignable(int &, void)'}} \
// expected-error@-1 {{assigning to 'int' from incompatible type 'void'}}

static_assert(__is_assignable(int*&, float*));
// expected-error@-1 {{static assertion failed due to requirement '__is_assignable(int *&, float *)'}} \
// expected-error@-1 {{incompatible pointer types assigning to 'int *' from 'float *'}}

static_assert(__is_assignable(const int&, int));
// expected-error@-1 {{static assertion failed due to requirement '__is_assignable(const int &, int)'}} \
// expected-error@-1 {{read-only variable is not assignable}}

struct S2 {}; // #a-S2
static_assert(__is_assignable(const S2, S2));
// expected-error@-1 {{static assertion failed due to requirement '__is_assignable(const assignable::S2, assignable::S2)'}} \
// expected-error@-1 {{no viable overloaded '='}} \
// expected-note@#a-S2 {{candidate function (the implicit copy assignment operator) not viable: 'this' argument has type 'const S2', but method is not marked const}} \
// expected-note@#a-S2 {{candidate function (the implicit move assignment operator) not viable: 'this' argument has type 'const S2', but method is not marked const}} \
// expected-note@#a-S2 {{'S2' defined here}}

struct S3 { // #a-S3
    S3& operator=(const S3&) = delete; // #aca-S3
    S3& operator=(S3&&) = delete;  // #ama-S3
};
static_assert(__is_assignable(S3, const S3&));
// expected-error@-1 {{static assertion failed due to requirement '__is_assignable(assignable::S3, const assignable::S3 &)'}} \
// expected-error@-1 {{overload resolution selected deleted operator '='}} \
// expected-note@#aca-S3 {{candidate function has been explicitly deleted}} \
// expected-note@#ama-S3 {{candidate function not viable: 1st argument ('const S3') would lose const qualifier}} \
// expected-note@#a-S3 {{'S3' defined here}}
static_assert(__is_assignable(S3, S3&&));
// expected-error@-1 {{static assertion failed due to requirement '__is_assignable(assignable::S3, assignable::S3 &&)'}} \
// expected-error@-1 {{overload resolution selected deleted operator '='}} \
// expected-note@#aca-S3 {{candidate function has been explicitly deleted}} \
// expected-note@#ama-S3 {{candidate function has been explicitly deleted}} \
// expected-note@#a-S3 {{'S3' defined here}}

class C1 { // #a-C1
    C1& operator=(const C1&) = default;
    C1& operator=(C1&&) = default; // #ama-C1
};
static_assert(__is_assignable(C1, C1));
// expected-error@-1 {{static assertion failed due to requirement '__is_assignable(assignable::C1, assignable::C1)'}} \
// expected-error@-1 {{'operator=' is a private member of 'assignable::C1'}} \
// expected-note@#ama-C1 {{implicitly declared private here}} \
// expected-note@#a-C1 {{'C1' defined here}}
}

namespace is_empty_tests {
    // Non-static data member.
    struct A { int x; }; // #e-A
    static_assert(__is_empty(A));
    // expected-error@-1 {{static assertion failed due to requirement '__is_empty(is_empty_tests::A)'}} \
    // expected-note@-1 {{'A' is not empty}} \
    // expected-note@-1 {{because it has a non-static data member 'x' of type 'int'}} \
    // expected-note@#e-A {{'A' defined here}}

    // Reference member.
    struct R {int &r; }; // #e-R
    static_assert(__is_empty(R));
    // expected-error@-1 {{static assertion failed due to requirement '__is_empty(is_empty_tests::R)'}} \
    // expected-note@-1 {{'R' is not empty}} \
    // expected-note@-1 {{because it has a non-static data member 'r' of type 'int &'}} \
    // expected-note@#e-R {{'R' defined here}}

    // Virtual function.
    struct VirtualFunc {virtual void f(); }; // #e-VirtualFunc
    static_assert(__is_empty(VirtualFunc));
    // expected-error@-1 {{static assertion failed due to requirement '__is_empty(is_empty_tests::VirtualFunc)'}} \
    // expected-note@-1 {{'VirtualFunc' is not empty}} \
    // expected-note@-1 {{because it has a virtual function 'f'}} \
    // expected-note@#e-VirtualFunc {{'VirtualFunc' defined here}}

    // Virtual base class.
    struct EB {};
    struct VB: virtual EB {}; // #e-VB
    static_assert(__is_empty(VB));
    // expected-error@-1 {{static assertion failed due to requirement '__is_empty(is_empty_tests::VB)'}} \
    // expected-note@-1 {{'VB' is not empty}} \
    // expected-note@-1 {{because it has a virtual base 'EB'}} \
    // expected-note@#e-VB {{'VB' defined here}}

    // Non-empty base class.
    struct Base { int b; }; // #e-Base
    struct Derived : Base {}; // #e-Derived
    static_assert(__is_empty(Derived));
    // expected-error@-1 {{static assertion failed due to requirement '__is_empty(is_empty_tests::Derived)'}} \
    // expected-note@-1 {{'Derived' is not empty}} \
    // expected-note@-1 {{because it has a base class 'Base' that is not empty}} \
    // expected-note@#e-Derived {{'Derived' defined here}} 

    // Combination of the above.
    struct Multi : Base, virtual EB { // #e-Multi
        int z;
        virtual void g();
    };
    static_assert(__is_empty(Multi));
    // expected-error@-1 {{static assertion failed due to requirement '__is_empty(is_empty_tests::Multi)'}} \
    // expected-note@-1 {{'Multi' is not empty}} \
    // expected-note@-1 {{because it has a non-static data member 'z' of type 'int'}} \
    // expected-note@-1 {{because it has a virtual function 'g'}} \
    // expected-note@-1 {{because it has a base class 'Base' that is not empty}} \
    // expected-note@-1 {{because it has a virtual base 'EB'}} \
    // expected-note@#e-Multi {{'Multi' defined here}}

    // Zero-width bit-field.
    struct BitField { int : 0; }; // #e-BitField
    static_assert(__is_empty(BitField)); // no diagnostics  

    // Dependent bit-field width. 
    template <int N>
    struct DependentBitField { int : N; }; // #e-DependentBitField

    static_assert(__is_empty(DependentBitField<0>)); // no diagnostics

    static_assert(__is_empty(DependentBitField<2>)); 
    // expected-error@-1 {{static assertion failed due to requirement '__is_empty(is_empty_tests::DependentBitField<2>)'}} \
    // expected-note@-1 {{'DependentBitField<2>' is not empty}} \
    // expected-note@-1 {{because it field '' is a non-zero-length bit-field}} \
    // expected-note@#e-DependentBitField {{'DependentBitField<2>' defined here}}

}

namespace standard_layout_tests {
struct WithVirtual { // #sl-Virtual
    virtual void foo(); // #sl-Virtual-Foo
};
static_assert(__is_standard_layout(WithVirtual));
// expected-error@-1 {{static assertion failed due to requirement '__is_standard_layout(standard_layout_tests::WithVirtual)'}} \
// expected-note@-1 {{'WithVirtual' is not standard-layout}} \
// expected-note@-1 {{because it has a virtual function 'foo'}} \
// expected-note@#sl-Virtual-Foo {{'foo' defined here}} \
// expected-note@#sl-Virtual {{'WithVirtual' defined here}}

struct MixedAccess { // #sl-Mixed
public:
    int a; // #sl-MixedF1
private:
    int b; // #sl-MixedF2
};
static_assert(__is_standard_layout(MixedAccess));
// expected-error@-1 {{static assertion failed due to requirement '__is_standard_layout(standard_layout_tests::MixedAccess)'}} \
// expected-note@-1 {{'MixedAccess' is not standard-layout}} \
// expected-note@-1 {{because it has mixed access specifiers}} \
// expected-note@#sl-MixedF1 {{'a' defined here}}
// expected-note@#sl-MixedF2 {{field 'b' has a different access specifier than field 'a'}}
// expected-note@#sl-Mixed {{'MixedAccess' defined here}}

struct VirtualBase { virtual ~VirtualBase(); };               // #sl-VirtualBase
struct VB : virtual VirtualBase {};                            // #sl-VB
static_assert(__is_standard_layout(VB));
// expected-error@-1 {{static assertion failed due to requirement '__is_standard_layout(standard_layout_tests::VB)'}} \
// expected-note@-1 {{'VB' is not standard-layout}} \
// expected-note@-1 {{because it has a virtual base 'VirtualBase'}} \
// expected-note@-1 {{because it has a non-standard-layout base 'VirtualBase'}} \
// expected-note@-1 {{because it has a virtual function '~VB'}} \
// expected-note@#sl-VB {{'VB' defined here}}
// expected-note@#sl-VB {{'~VB' defined here}}

union U {      // #sl-U
public:
    int x; // #sl-UF1
private:
    int y; // #sl-UF2
};                                                       
static_assert(__is_standard_layout(U));
// expected-error@-1 {{static assertion failed due to requirement '__is_standard_layout(standard_layout_tests::U)'}} \
// expected-note@-1 {{'U' is not standard-layout}} \
// expected-note@-1 {{because it has mixed access specifiers}}
// expected-note@#sl-UF1 {{'x' defined here}}
// expected-note@#sl-UF2 {{field 'y' has a different access specifier than field 'x'}}
// expected-note@#sl-U {{'U' defined here}}

// Single base class is OK
struct BaseClass{ int a; };                                   // #sl-BaseClass
struct DerivedOK : BaseClass {};                                // #sl-DerivedOK
static_assert(__is_standard_layout(DerivedOK));    

// Primitive types should be standard layout
static_assert(__is_standard_layout(int));                     // #sl-Int
static_assert(__is_standard_layout(float));                   // #sl-Float

// Multi-level inheritance: Non-standard layout
struct Base1 { int a; };                                      // #sl-Base1
struct Base2 { int b; };                                      // #sl-Base2
struct DerivedClass : Base1, Base2 {};                        // #sl-DerivedClass
static_assert(__is_standard_layout(DerivedClass));               
// expected-error@-1 {{static assertion failed due to requirement '__is_standard_layout(standard_layout_tests::DerivedClass)'}} \
// expected-note@-1 {{'DerivedClass' is not standard-layout}} \
// expected-note@-1 {{because it has multiple base classes with data members}} \
// expected-note@#sl-DerivedClass {{'DerivedClass' defined here}} 

// Inheritance hierarchy with multiple classes having data members
struct BaseA { int a; };                                      // #sl-BaseA
struct BaseB : BaseA {};                                      // inherits BaseA, has no new members
struct BaseC: BaseB { int c; };                               // #sl-BaseC
static_assert(__is_standard_layout(BaseC));
// expected-error@-1 {{static assertion failed due to requirement '__is_standard_layout(standard_layout_tests::BaseC)'}} \
// expected-note@-1 {{'BaseC' is not standard-layout}} \
// expected-note@-1 {{because it has an indirect base 'BaseA' with data members}} \
// expected-note@#sl-BaseC {{'BaseC' defined here}} \
// Multiple direct base classes with no data members --> standard layout
struct BaseX {};                                              // #sl-BaseX
struct BaseY {};                                              // #sl-BaseY
struct MultiBase : BaseX, BaseY {};                          // #sl-MultiBase
static_assert(__is_standard_layout(MultiBase));

struct A {
  int x;
};

struct B : A {
};
// Indirect base with data members
struct C : B { int y; }; // #sl-C
static_assert(__is_standard_layout(C));
// expected-error@-1 {{static assertion failed due to requirement '__is_standard_layout(standard_layout_tests::C)'}} \
// expected-note@-1 {{'C' is not standard-layout}} \
// expected-note@-1 {{because it has an indirect base 'A' with data members}} \
// expected-note@#sl-C {{'C' defined here}}

struct D {
    union { int a; float b; };
  }; // #sl-D
static_assert(__is_standard_layout(D)); // no diagnostics

// E inherits D but adds a new member
struct E : D { int x; }; // #sl-E
static_assert(__is_standard_layout(E));
// expected-error@-1 {{static assertion failed due to requirement '__is_standard_layout(standard_layout_tests::E)'}} \
// expected-note@-1 {{'E' is not standard-layout}} \
// expected-note@-1 {{because it has an indirect base 'D' with data members}} \
// expected-note@#sl-E {{'E' defined here}}

// F inherits D but only an unnamed bitfield
// This should still fail because F ends up with a 
// base class with a data member and its own unnamed bitfield
// which is not allowed in standard layout
struct F : D { int : 0; }; // #sl-F
static_assert(__is_standard_layout(F));
// expected-error@-1 {{static assertion failed due to requirement '__is_standard_layout(standard_layout_tests::F)'}} \
// expected-note@-1 {{'F' is not standard-layout}} \
// expected-note@#sl-F {{'F' defined here}}

struct Empty {};
struct G { Empty a, b; }; // #sl-G
static_assert(__is_standard_layout(G)); // no diagnostics

struct H { Empty a; int x; }; // #sl-H
static_assert(__is_standard_layout(H)); // no diagnostics

 struct I { Empty a; int : 0; int x; }; // #sl-I
static_assert(__is_standard_layout(I)); // no diagnostics
}

namespace is_final_tests {
    struct C {}; // #e-C
    static_assert(__is_final(C));
    // expected-error@-1 {{static assertion failed due to requirement '__is_final(is_final_tests::C)'}} \
    // expected-note@-1 {{'C' is not final}} \
    // expected-note@-1 {{because it is not marked 'final'}} \
    // expected-note@#e-C {{'C' defined here}}

    union U {}; // #e-U
    static_assert(__is_final(U));
    // expected-error@-1 {{static assertion failed due to requirement '__is_final(is_final_tests::U)'}} \
    // expected-note@-1 {{'U' is not final}} \
    // expected-note@-1 {{because it is not marked 'final'}} \
    // expected-note@#e-U {{'U' defined here}}

    // ----- non-class/union types -----
    using I = int;
    static_assert(__is_final(I));
    // expected-error@-1 {{static assertion failed due to requirement '__is_final(int)'}} \
    // expected-note@-1 {{'I' (aka 'int') is not final}} \
    // expected-note@-1 {{because it is not a class or union type}}

    using Fty = void(); // function type
    static_assert(__is_final(Fty));
    // expected-error@-1 {{static assertion failed due to requirement '__is_final(void ())'}} \
    // expected-note@-1 {{'Fty' (aka 'void ()') is not final}} \
    // expected-note@-1 {{because it is a function type}} \
    // expected-note@-1 {{because it is not a class or union type}}

    using Arr = int[3];
    static_assert(__is_final(Arr));
    // expected-error@-1 {{static assertion failed due to requirement '__is_final(int[3])'}} \
    // expected-note@-1 {{'Arr' (aka 'int[3]') is not final}} \
    // expected-note@-1 {{because it is not a class or union type}}

    using Ref = int&;
    static_assert(__is_final(Ref));
    // expected-error@-1 {{static assertion failed due to requirement '__is_final(int &)'}} \
    // expected-note@-1 {{'Ref' (aka 'int &') is not final}} \
    // expected-note@-1 {{because it is a reference type}} \
    // expected-note@-1 {{because it is not a class or union type}}

}
