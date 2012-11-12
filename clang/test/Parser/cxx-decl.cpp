// RUN: %clang_cc1 -verify -fsyntax-only -triple i386-linux -pedantic %s

const char const *x10; // expected-warning {{duplicate 'const' declaration specifier}}

int x(*g); // expected-error {{use of undeclared identifier 'g'}}

struct Type {
  int Type;
};

// rdar://8365458
// rdar://9132143
typedef char bool; // expected-error {{redeclaration of C++ built-in type 'bool'}}

// PR4451 - We should recover well from the typo of '::' as ':' in a2.
namespace y {
  struct a { };
  typedef int b;
}

y::a a1;
y:a a2;  // expected-error {{unexpected ':' in nested name specifier}}
y::a a3 = a2;

// Some valid colons:
void foo() {
y:  // label
  y::a s;
  
  int a = 4;
  a = a ? a : a+1;
}

struct b : y::a {};

template <typename T>
class someclass {
  
  int bar() {
    T *P;
    return 1 ? P->x : P->y;
  }
};

class asm_class_test {
  void foo() __asm__("baz");
};

enum { fooenum = 1, }; // expected-warning {{commas at the end of enumerator lists are a C++11 extension}}

struct a {
  int Type : fooenum;
};

void test(struct Type *P) {
  int Type;
  Type = 1 ? P->Type : Type;
  
  Type = (y:b) 4;   // expected-error {{unexpected ':' in nested name specifier}}
  Type = 1 ? (
              (y:b)  // expected-error {{unexpected ':' in nested name specifier}}
              4) : 5;
}

struct test4 {
  int x  // expected-error {{expected ';' at end of declaration list}}
  int y;
  int z  // expected-error {{expected ';' at end of declaration list}}
};

// Make sure we know these are legitimate commas and not typos for ';'.
namespace Commas {
  struct S {
    static int a;
    int c,
    operator()();
  };

  int global1,
  __attribute__(()) global2,
  (global5),
  *global6,
  &global7 = global1,
  &&global8 = static_cast<int&&>(global1), // expected-warning 2{{rvalue reference}}
  S::a,
  global9,
  global10 = 0,
  global11 == 0, // expected-error {{did you mean '='}}
  global12 __attribute__(()),
  global13(0),
  global14[2],
  global15;

  void g() {
    static int a,
    b __asm__("ebx"), // expected-error {{expected ';' at end of declaration}}
    Statics:return;
  }
}

// PR5825
struct test5 {};
::new(static_cast<void*>(0)) test5; // expected-error {{expected unqualified-id}}


// PR6782
template<class T>
class Class1;

class Class2 {
}  // no ;

typedef Class1<Class2> Type1; // expected-error {{cannot combine with previous 'class' declaration specifier}}

// rdar : // 8307865
struct CodeCompleteConsumer {
};

void CodeCompleteConsumer::() { // expected-error {{xpected unqualified-id}}
} 

;

// PR4111
void f(sqrgl); // expected-error {{unknown type name 'sqrgl'}}

// PR8380
extern ""      // expected-error {{unknown linkage language}}
test6a { ;// expected-error {{C++ requires a type specifier for all declarations}} \
     // expected-error {{expected ';' after top level declarator}}
  
  int test6b;
