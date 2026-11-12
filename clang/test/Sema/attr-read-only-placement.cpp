// RUN: %clang_cc1 -Wread-only-types %s -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++2a -Wread-only-types %s -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++17 -Wread-only-types %s -verify -fsyntax-only

struct __attribute__((enforce_read_only_placement)) A { // #A_DECL
};

A a1; // expected-warning {{object of type 'A' cannot be placed in read-only memory}}
      // expected-note@#A_DECL {{type was declared read-only here}}
const A a2[10]; // no-warning
A a3[20]; // expected-warning {{object of type 'A' cannot be placed in read-only memory}}
          // expected-note@#A_DECL {{type was declared read-only here}}



struct B;
struct __attribute__((enforce_read_only_placement)) B { //#B_DECL
};

B b1; // expected-warning {{object of type 'B' cannot be placed in read-only memory}}
      // expected-note@#B_DECL {{type was declared read-only here}}
const B b2; // no-warning
const B b3[4]; // no-warning
B b4[5]; // expected-warning {{object of type 'B' cannot be placed in read-only memory}}
         // expected-note@#B_DECL {{type was declared read-only here}}
B b5[5][5]; // expected-warning {{object of type 'B' cannot be placed in read-only memory}}
            // expected-note@#B_DECL {{type was declared read-only here}}
B b10[5][5][5]; // expected-warning {{object of type 'B' cannot be placed in read-only memory}}
                // expected-note@#B_DECL {{type was declared read-only here}}

void method1() {
    static const B b6;
    static B b7;// expected-warning {{object of type 'B' cannot be placed in read-only memory}}
                // expected-note@#B_DECL {{type was declared read-only here}}
    B b8; // no-warning
    const B b9; // no-warning
}

struct C;
struct __attribute__((enforce_read_only_placement)) C; // expected-note {{type was declared read-only here}}
struct C { // no-note. The note should be attached to the definition/declaration bearing the attribute
};

C c1; // expected-warning {{object of type 'C' cannot be placed in read-only memory}}

// Cases to be handled by the follow-up patches.

// Attaching and checking the attribute in reverse, where the attribute is attached after the
// type definition
struct D;
struct D { //expected-note{{previous definition is here}}
};
struct __attribute__((enforce_read_only_placement)) D; // #3
                // expected-warning@#3{{attribute declaration must precede definition}}

D d1; // We do not  emit a warning here, as there is another warning for declaring
      // a type after the definition


// Cases where the attribute must be explicitly attached to another type
// Case 1: Inheriting from a type that has the attribute
struct E : C { // FIXME: warn the user declarations of type `E`, that extends `C`, won't be
               // checked for read only placement because `E` is not marked as `C` is.
};

// Case 2: Declaring a field of the type that has the attribute
struct F {
    C c1; // FIXME: warn the user type `F` that wraps type `C` won't be checked for
          // read only placement
};

struct BaseWithoutAttribute {
    int a;
};

struct  __attribute__((enforce_read_only_placement)) J : BaseWithoutAttribute { // no-warning
};

struct __attribute__((enforce_read_only_placement)) BaseWithAttribute {
    int i;
};

struct __attribute__((enforce_read_only_placement)) Derived : BaseWithAttribute { // no-warning
    int j;
};

struct __attribute__((enforce_read_only_placement)) WrapperToAttributeInstance { // no-warning
    BaseWithAttribute b;
};

struct __attribute__((enforce_read_only_placement)) WrapperToNoAttributeInstance { // no-warning
    BaseWithoutAttribute b;
};

// Cases where the const qualification doesn't ensure read-only memory placement
// of an instance.

// Case 1: The type defines/inherits mutable data members
struct __attribute__((enforce_read_only_placement)) G {
    mutable int x; // FIXME: warn the user type `G` won't be placed in the read only program memory
};

struct __attribute__((enforce_read_only_placement)) H : public G { // FIXME: Warn the user type `H`
                                                // won't be placed in the read only program memory
};

struct __attribute__((enforce_read_only_placement)) K { // FIXME : Warn the user type `K` w on't be
                                                // placed in the read only program memory
    G g;
};


// Case 2: The type has a constructor that makes its fields modifiable
struct  __attribute__((enforce_read_only_placement)) L {
    int b;
    L(int val) { // FIXME: warn the user type `L` won't be placed in the read only program memory
      b = val;
    }
};

struct __attribute__((enforce_read_only_placement)) ConstInClassInitializers { // no-warning
  int b = 12;

  ConstInClassInitializers() = default;
};

int foo();
struct __attribute__((enforce_read_only_placement)) NonConstInClassInitializers {
  int b = foo(); // FIXME: warn the user type `NonConstInClassInitializers` won't be placed
                 // in the read only program memory

  NonConstInClassInitializers() = default;
};

#if (__cplusplus >= 202002L)
struct __attribute__((enforce_read_only_placement)) ConstevalCtor {
  int b;

  consteval ConstevalCtor(int B) : b(B) {} // no-warning
};
#endif

#if (__cplusplus >= 201103L)
struct __attribute__((enforce_read_only_placement)) ConstExprCtor { // no-warning
  int b;

  constexpr ConstExprCtor(int B) : b(B) {}
};

constexpr ConstExprCtor cec1(10); // no-warning

#endif

// Cases where an object is allocated on the heap or on the stack
C *c2 = new C; // FIXME: warn the user this instance of 'C' won't be placed in the read only program memory

void func1(C c); // FIXME: warn the user the instance of 'C' won't be placed in the read only program memory

void func2(const C c); // FIXME: warn the user the instance of 'C' won't be placed in the read
                       // only program memory

C func3(); // FIXME: warn the user the instance of 'C' won't be placed in the read only program memory

void func4() {
    C c; // FIXME: warn the user the instance of 'C' won't be placed in the read only program memory
}

#if (__cplusplus >= 202002L)
consteval void func4(C c); // no-warning
#endif
