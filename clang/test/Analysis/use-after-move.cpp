// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.Move -verify %s\
// RUN:  -std=c++11 -analyzer-output=text -analyzer-config eagerly-assume=false\
// RUN:  -analyzer-config exploration_strategy=unexplored_first_queue\
// RUN:  -analyzer-checker debug.ExprInspection
// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.Move -verify %s\
// RUN:  -std=c++11 -analyzer-output=text -analyzer-config eagerly-assume=false\
// RUN:  -analyzer-config exploration_strategy=dfs -DDFS=1\
// RUN:  -analyzer-checker debug.ExprInspection
// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.Move -verify %s\
// RUN:  -std=c++11 -analyzer-output=text -analyzer-config eagerly-assume=false\
// RUN:  -analyzer-config exploration_strategy=unexplored_first_queue\
// RUN:  -analyzer-config cplusplus.Move:WarnOn=KnownsOnly -DPEACEFUL\
// RUN:  -analyzer-checker debug.ExprInspection
// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.Move -verify %s\
// RUN:  -std=c++11 -analyzer-output=text -analyzer-config eagerly-assume=false\
// RUN:  -analyzer-config exploration_strategy=dfs -DDFS=1\
// RUN:  -analyzer-config cplusplus.Move:WarnOn=KnownsOnly -DPEACEFUL\
// RUN:  -analyzer-checker debug.ExprInspection
// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.Move -verify %s\
// RUN:  -std=c++11 -analyzer-output=text -analyzer-config eagerly-assume=false\
// RUN:  -analyzer-config exploration_strategy=unexplored_first_queue\
// RUN:  -analyzer-config cplusplus.Move:WarnOn=All -DAGGRESSIVE\
// RUN:  -analyzer-checker debug.ExprInspection
// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.Move -verify %s\
// RUN:  -std=c++11 -analyzer-output=text -analyzer-config eagerly-assume=false\
// RUN:  -analyzer-config exploration_strategy=dfs -DDFS=1\
// RUN:  -analyzer-config cplusplus.Move:WarnOn=All -DAGGRESSIVE\
// RUN:  -analyzer-checker debug.ExprInspection

// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.Move \
// RUN:   -analyzer-config cplusplus.Move:WarnOn="a bunch of things" \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-MOVE-INVALID-VALUE

// CHECK-MOVE-INVALID-VALUE: (frontend): invalid input for checker option
// CHECK-MOVE-INVALID-VALUE-SAME: 'cplusplus.Move:WarnOn', that expects either
// CHECK-MOVE-INVALID-VALUE-SAME: "KnownsOnly", "KnownsAndLocals" or "All"
// CHECK-MOVE-INVALID-VALUE-SAME: string value

#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_warnIfReached();

class B {
public:
  B() = default;
  B(const B &) = default;
  B(B &&) = default;
  B& operator=(const B &q) = default;
  void operator=(B &&b) {
    return;
  }
  void foo() { return; }
};

class A {
  int i;
  double d;

public:
  B b;
  A(int ii = 42, double dd = 1.0) : d(dd), i(ii), b(B()) {}
  void moveconstruct(A &&other) {
    std::swap(b, other.b);
    std::swap(d, other.d);
    std::swap(i, other.i);
    return;
  }
  static A get() {
    A v(12, 13);
    return v;
  }
  A(A *a) {
    moveconstruct(std::move(*a));
  }
  A(const A &other) : i(other.i), d(other.d), b(other.b) {}
  A(A &&other) : i(other.i), d(other.d), b(std::move(other.b)) {
#ifdef AGGRESSIVE
    // expected-note@-2{{Object 'b' is moved}}
#endif
  }
  A(A &&other, char *k) {
    moveconstruct(std::move(other));
  }
  void operator=(const A &other) {
    i = other.i;
    d = other.d;
    b = other.b;
    return;
  }
  void operator=(A &&other) {
    moveconstruct(std::move(other));
    return;
  }
  int getI() { return i; }
  int foo() const;
  void bar() const;
  void reset();
  void destroy();
  void clear();
  void resize(std::size_t);
  void assign(const A &);
  bool empty() const;
  bool isEmpty() const;
  operator bool() const;

  void testUpdateField() {
    A a;
    A b = std::move(a);
    a.i = 1;
    a.foo(); // no-warning
  }
  void testUpdateFieldDouble() {
    A a;
    A b = std::move(a);
    a.d = 1.0;
    a.foo(); // no-warning
  }
};

int bignum();

void moveInsideFunctionCall(A a) {
  A b = std::move(a);
}
void leftRefCall(A &a) {
  a.foo();
}
void rightRefCall(A &&a) {
  a.foo();
}
void constCopyOrMoveCall(const A a) {
  a.foo();
}

void copyOrMoveCall(A a) {
  a.foo();
}

void simpleMoveCtorTest() {
  {
    A a;
    A b = std::move(a);
    a.foo();
#ifndef PEACEFUL
    // expected-note@-3 {{Object 'a' is moved}}
    // expected-warning@-3 {{Method called on moved-from object 'a'}}
    // expected-note@-4    {{Method called on moved-from object 'a'}}
#endif
  }
  {
    A a;
    A b = std::move(a);
    b = a;
#ifndef PEACEFUL
    // expected-note@-3 {{Object 'a' is moved}}
    // expected-warning@-3 {{Moved-from object 'a' is copied}}
    // expected-note@-4    {{Moved-from object 'a' is copied}}
#endif
  }
  {
    A a;
    A b = std::move(a);
    b = std::move(a);
#ifndef PEACEFUL
    // expected-note@-3 {{Object 'a' is moved}}
    // expected-warning@-3 {{Moved-from object 'a' is moved}}
    // expected-note@-4    {{Moved-from object 'a' is moved}}
#endif
  }
}

void simpleMoveAssignementTest() {
  {
    A a;
    A b;
    b = std::move(a);
    a.foo();
#ifndef PEACEFUL
    // expected-note@-3 {{Object 'a' is moved}}
    // expected-warning@-3 {{Method called on moved-from object 'a'}}
    // expected-note@-4    {{Method called on moved-from object 'a'}}
#endif
  }
  {
    A a;
    A b;
    b = std::move(a);
    A c(a);
#ifndef PEACEFUL
    // expected-note@-3 {{Object 'a' is moved}}
    // expected-warning@-3 {{Moved-from object 'a' is copied}}
    // expected-note@-4    {{Moved-from object 'a' is copied}}
#endif
  }
  {
    A a;
    A b;
    b = std::move(a);
    A c(std::move(a));
#ifndef PEACEFUL
    // expected-note@-3 {{Object 'a' is moved}}
    // expected-warning@-3 {{Moved-from object 'a' is moved}}
    // expected-note@-4    {{Moved-from object 'a' is moved}}
#endif
  }
}

void moveInInitListTest() {
  struct S {
    A a;
  };
  A a;
  S s{std::move(a)};
  a.foo();
#ifndef PEACEFUL
  // expected-note@-3 {{Object 'a' is moved}}
  // expected-warning@-3 {{Method called on moved-from object 'a'}}
  // expected-note@-4 {{Method called on moved-from object 'a'}}
#endif
}

// Don't report a bug if the variable was assigned to in the meantime.
void reinitializationTest(int i) {
  {
    A a;
    A b;
    b = std::move(a);
    a = A();
    a.foo();
  }
  {
    A a;
    if (i == 1) {
#ifndef PEACEFUL
      // expected-note@-2 {{Assuming 'i' is not equal to 1}}
      // expected-note@-3 {{Taking false branch}}
      // And the other report:
      // expected-note@-5 {{Assuming 'i' is not equal to 1}}
      // expected-note@-6 {{Taking false branch}}
#endif
      A b;
      b = std::move(a);
      a = A();
    }
    if (i == 2) {
#ifndef PEACEFUL
      // expected-note@-2 {{Assuming 'i' is not equal to 2}}
      // expected-note@-3 {{Taking false branch}}
      // And the other report:
      // expected-note@-5 {{Assuming 'i' is not equal to 2}}
      // expected-note@-6 {{Taking false branch}}
#endif
      a.foo();    // no-warning
    }
  }
  {
    A a;
    if (i == 1) {
#ifndef PEACEFUL
      // expected-note@-2 {{Taking false branch}}
      // expected-note@-3 {{Taking false branch}}
#endif
      std::move(a);
    }
    if (i == 2) {
#ifndef PEACEFUL
      // expected-note@-2 {{Taking false branch}}
      // expected-note@-3 {{Taking false branch}}
#endif
      a = A();
      a.foo();
    }
  }
  // The built-in assignment operator should also be recognized as a
  // reinitialization. (std::move() may be called on built-in types in template
  // code.)
  {
    int a1 = 1, a2 = 2;
    std::swap(a1, a2);
  }
  // A std::move() after the assignment makes the variable invalid again.
  {
    A a;
    A b;
    b = std::move(a);
    a = A();
    b = std::move(a);
    a.foo();
#ifndef PEACEFUL
    // expected-note@-3 {{Object 'a' is moved}}
    // expected-warning@-3 {{Method called on moved-from object 'a'}}
    // expected-note@-4    {{Method called on moved-from object 'a'}}
#endif
  }
  // If a path exist where we not reinitialize the variable we report a bug.
  {
    A a;
    A b;
    b = std::move(a);
#ifndef PEACEFUL
    // expected-note@-2 {{Object 'a' is moved}}
#endif
    if (i < 10) {
#ifndef PEACEFUL
      // expected-note@-2 {{Assuming 'i' is >= 10}}
      // expected-note@-3 {{Taking false branch}}
#endif
      a = A();
    }
    if (i > 5) {
      a.foo();
#ifndef PEACEFUL
      // expected-note@-3 {{Taking true branch}}
      // expected-warning@-3 {{Method called on moved-from object 'a'}}
      // expected-note@-4    {{Method called on moved-from object 'a'}}
#endif
    }
  }
}

// Using decltype on an expression is not a use.
void decltypeIsNotUseTest() {
  A a;
  // A b(std::move(a));
  decltype(a) other_a; // no-warning
}

void loopTest() {
  {
    A a;
    for (int i = 0; i < bignum(); i++) {
#ifndef PEACEFUL
      // expected-note@-2 {{Loop condition is false. Execution jumps to the end of the function}}
#endif
      rightRefCall(std::move(a));        // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < 2; i++) {
#ifndef PEACEFUL
      // expected-note@-2 {{Loop condition is true.  Entering loop body}}
      // expected-note@-3 {{Loop condition is true.  Entering loop body}}
      // expected-note@-4 {{Loop condition is false. Execution jumps to the end of the function}}
#endif
      rightRefCall(std::move(a)); // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < bignum(); i++) {
#ifndef PEACEFUL
      // expected-note@-2 {{Loop condition is false. Execution jumps to the end of the function}}
#endif
      leftRefCall(a);                    // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < 2; i++) {
#ifndef PEACEFUL
      // expected-note@-2 {{Loop condition is true.  Entering loop body}}
      // expected-note@-3 {{Loop condition is true.  Entering loop body}}
      // expected-note@-4 {{Loop condition is false. Execution jumps to the end of the function}}
#endif
      leftRefCall(a);             // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < bignum(); i++) {
#ifndef PEACEFUL
      // expected-note@-2 {{Loop condition is false. Execution jumps to the end of the function}}
#endif
      constCopyOrMoveCall(a);            // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < 2; i++) {
#ifndef PEACEFUL
      // expected-note@-2 {{Loop condition is true.  Entering loop body}}
      // expected-note@-3 {{Loop condition is true.  Entering loop body}}
      // expected-note@-4 {{Loop condition is false. Execution jumps to the end of the function}}
#endif
      constCopyOrMoveCall(a);     // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < bignum(); i++) {
#ifndef PEACEFUL
      // expected-note@-2 {{Loop condition is false. Execution jumps to the end of the function}}
#endif
      moveInsideFunctionCall(a);         // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < 2; i++) {
#ifndef PEACEFUL
      // expected-note@-2 {{Loop condition is true.  Entering loop body}}
      // expected-note@-3 {{Loop condition is true.  Entering loop body}}
      // expected-note@-4 {{Loop condition is false. Execution jumps to the end of the function}}
#endif
      moveInsideFunctionCall(a);  // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < bignum(); i++) {
#ifndef PEACEFUL
      // expected-note@-2 {{Loop condition is false. Execution jumps to the end of the function}}
#endif
      copyOrMoveCall(a);                 // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < 2; i++) {
#ifndef PEACEFUL
      // expected-note@-2 {{Loop condition is true.  Entering loop body}}
      // expected-note@-3 {{Loop condition is true.  Entering loop body}}
      // expected-note@-4 {{Loop condition is false. Execution jumps to the end of the function}}
#endif
      copyOrMoveCall(a);          // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < bignum(); i++) {
#ifndef PEACEFUL
      // expected-note@-2 {{Loop condition is true.  Entering loop body}}
      // expected-note@-3 {{Loop condition is true.  Entering loop body}}
#endif
      constCopyOrMoveCall(std::move(a));
#ifndef PEACEFUL
      // expected-note@-2    {{Object 'a' is moved}}
      // expected-warning@-3 {{Moved-from object 'a' is moved}}
      // expected-note@-4 {{Moved-from object 'a' is moved}}
#endif
    }
  }

  // Don't warn if we return after the move.
  {
    A a;
    for (int i = 0; i < 3; ++i) {
      a.bar();
      if (a.foo() > 0) {
        A b;
        b = std::move(a); // no-warning
        return;
      }
    }
  }
}

// Report a usage of a moved-from object only at the first use.
void uniqueTest(bool cond) {
  A a(42, 42.0);
  A b;
  b = std::move(a);

  if (cond) {
    a.foo();
#ifndef PEACEFUL
  // expected-note@-5 {{Object 'a' is moved}}
  // expected-note@-4 {{Assuming 'cond' is not equal to 0}}
  // expected-note@-5 {{Taking true branch}}
  // expected-warning@-5 {{Method called on moved-from object 'a'}}
  // expected-note@-6    {{Method called on moved-from object 'a'}}
#endif
  }
  if (cond) {
    a.bar(); // no-warning
  }

  a.bar(); // no-warning
}

void uniqueTest2() {
  A a;
  A a1 = std::move(a);
  a.foo();
#ifndef PEACEFUL
  // expected-note@-3 {{Object 'a' is moved}}
  // expected-warning@-3 {{Method called on moved-from object 'a'}}
  // expected-note@-4    {{Method called on moved-from object 'a'}}
#endif

  A a2 = std::move(a); // no-warning
  a.foo();             // no-warning
}

// There are exceptions where we assume in general that the method works fine
//even on moved-from objects.
void moveSafeFunctionsTest() {
  A a;
  A b = std::move(a);
#ifndef PEACEFUL
  // expected-note@-2 {{Object 'a' is moved}}
#endif
  a.empty();          // no-warning
  a.isEmpty();        // no-warning
  (void)a;            // no-warning
  (bool)a;            // expected-warning {{expression result unused}}
  a.foo();
#ifndef PEACEFUL
  // expected-warning@-2 {{Method called on moved-from object 'a'}}
  // expected-note@-3    {{Method called on moved-from object 'a'}}
#endif
}

void moveStateResetFunctionsTest() {
  {
    A a;
    A b = std::move(a);
    a.reset(); // no-warning
    a.foo();   // no-warning
    // Test if resets the state of subregions as well.
    a.b.foo(); // no-warning
  }
  {
    A a;
    A b = std::move(a);
    a.destroy(); // no-warning
    a.foo();     // no-warning
  }
  {
    A a;
    A b = std::move(a);
    a.clear(); // no-warning
    a.foo();   // no-warning
    a.b.foo(); // no-warning
  }
  {
    A a;
    A b = std::move(a);
    a.resize(0); // no-warning
    a.foo();   // no-warning
    a.b.foo(); // no-warning
  }
  {
    A a;
    A b = std::move(a);
    a.assign(A()); // no-warning
    a.foo();   // no-warning
    a.b.foo(); // no-warning
  }
}

// Moves or uses that occur as part of template arguments.
template <int>
class ClassTemplate {
public:
  void foo(A a);
};

template <int>
void functionTemplate(A a);

void templateArgIsNotUseTest() {
  {
    // A pattern like this occurs in the EXPECT_EQ and ASSERT_EQ macros in
    // Google Test.
    A a;
    ClassTemplate<sizeof(A(std::move(a)))>().foo(std::move(a)); // no-warning
  }
  {
    A a;
    functionTemplate<sizeof(A(std::move(a)))>(std::move(a)); // no-warning
  }
}

// Moves of global variables are not reported.
A global_a;
void globalVariablesTest() {
  std::move(global_a);
  global_a.foo(); // no-warning
}

// Moves of member variables.
class memberVariablesTest {
  A a;
  static A static_a;

  void f() {
    A b;
    b = std::move(a);
    a.foo();
#ifdef AGGRESSIVE
    // expected-note@-3{{Object 'a' is moved}}
    // expected-warning@-3 {{Method called on moved-from object 'a'}}
    // expected-note@-4{{Method called on moved-from object 'a'}}
#endif

    b = std::move(static_a);
    static_a.foo();
#ifdef AGGRESSIVE
    // expected-note@-3{{Object 'static_a' is moved}}
    // expected-warning@-3{{Method called on moved-from object 'static_a'}}
    // expected-note@-4{{Method called on moved-from object 'static_a'}}
#endif
  }
};

void PtrAndArrayTest() {
  A *Ptr = new A(1, 1.5);
  A Arr[10];
  Arr[2] = std::move(*Ptr);
  (*Ptr).foo();
#ifdef AGGRESSIVE
  // expected-note@-3{{Object is moved}}
  // expected-warning@-3{{Method called on moved-from object}}
  // expected-note@-4{{Method called on moved-from object}}
#endif

  Ptr = &Arr[1];
  Arr[3] = std::move(Arr[1]);
  Ptr->foo();
#ifdef AGGRESSIVE
  // expected-note@-3{{Object is moved}}
  // expected-warning@-3{{Method called on moved-from object}}
  // expected-note@-4{{Method called on moved-from object}}
#endif

  Arr[3] = std::move(Arr[2]);
  Arr[2].foo();
#ifdef AGGRESSIVE
  // expected-note@-3{{Object is moved}}
  // expected-warning@-3{{Method called on moved-from object}}
  // expected-note@-4{{Method called on moved-from object}}
#endif

  Arr[2] = std::move(Arr[3]); // reinitialization
  Arr[2].foo();               // no-warning
}

void exclusiveConditionsTest(bool cond) {
  A a;
  if (cond) {
    A b;
    b = std::move(a);
  }
  if (!cond) {
    a.bar(); // no-warning
  }
}

void differentBranchesTest(int i) {
  // Don't warn if the use is in a different branch from the move.
  {
    A a;
    if (i > 0) {
#ifndef PEACEFUL
    // expected-note@-2 {{Assuming 'i' is > 0}}
    // expected-note@-3 {{Taking true branch}}
#endif
      A b;
      b = std::move(a);
    } else {
      a.foo(); // no-warning
    }
  }
  // Same thing, but with a ternary operator.
  {
    A a, b;
    i > 0 ? (void)(b = std::move(a)) : a.bar(); // no-warning
#ifndef PEACEFUL
    // expected-note@-2 {{'?' condition is true}}
#endif
  }
  // A variation on the theme above.
  {
    A a;
    a.foo() > 0 ? a.foo() : A(std::move(a)).foo();
#ifdef DFS
  #ifndef PEACEFUL
    // expected-note@-3 {{Assuming the condition is false}}
    // expected-note@-4 {{'?' condition is false}}
  #endif
#else
  #ifndef PEACEFUL
    // expected-note@-8 {{Assuming the condition is true}}
    // expected-note@-9 {{'?' condition is true}}
  #endif
#endif
  }
  // Same thing, but with a switch statement.
  {
    A a, b;
    switch (i) {
#ifndef PEACEFUL
    // expected-note@-2 {{Control jumps to 'case 1:'}}
#endif
    case 1:
      b = std::move(a); // no-warning
      break;
#ifndef PEACEFUL
      // expected-note@-2 {{Execution jumps to the end of the function}}
#endif
    case 2:
      a.foo(); // no-warning
      break;
    }
  }
  // However, if there's a fallthrough, we do warn.
  {
    A a, b;
    switch (i) {
#ifndef PEACEFUL
    // expected-note@-2 {{Control jumps to 'case 1:'}}
#endif
    case 1:
      b = std::move(a);
#ifndef PEACEFUL
      // expected-note@-2 {{Object 'a' is moved}}
#endif
    case 2:
      a.foo();
#ifndef PEACEFUL
      // expected-warning@-2 {{Method called on moved-from object}}
      // expected-note@-3    {{Method called on moved-from object 'a'}}
#endif
      break;
    }
  }
}

void tempTest() {
  A a = A::get();
  A::get().foo(); // no-warning
  for (int i = 0; i < bignum(); i++) {
    A::get().foo(); // no-warning
  }
}

void interFunTest1(A &a) {
  a.bar();
#ifndef PEACEFUL
  // expected-warning@-2 {{Method called on moved-from object 'a'}}
  // expected-note@-3    {{Method called on moved-from object 'a'}}
#endif
}

void interFunTest2() {
  A a;
  A b;
  b = std::move(a);
  interFunTest1(a);
#ifndef PEACEFUL
  // expected-note@-3 {{Object 'a' is moved}}
  // expected-note@-3 {{Calling 'interFunTest1'}}
#endif
}

void foobar(A a, int i);
void foobar(int i, A a);

void paramEvaluateOrderTest() {
  A a;
  foobar(std::move(a), a.getI());
#ifndef PEACEFUL
  // expected-note@-2 {{Object 'a' is moved}}
  // expected-warning@-3 {{Method called on moved-from object 'a'}}
  // expected-note@-4    {{Method called on moved-from object 'a'}}
#endif

  //FALSE NEGATIVE since parameters evaluate order is undefined
  foobar(a.getI(), std::move(a)); //no-warning
}

void not_known_pass_by_ref(A &a);
void not_known_pass_by_const_ref(const A &a);
void not_known_pass_by_rvalue_ref(A &&a);
void not_known_pass_by_ptr(A *a);
void not_known_pass_by_const_ptr(const A *a);

void regionAndPointerEscapeTest() {
  {
    A a;
    A b;
    b = std::move(a);
    not_known_pass_by_ref(a);
    a.foo(); // no-warning
  }
  {
    A a;
    A b;
    b = std::move(a);
    not_known_pass_by_const_ref(a);
    a.foo();
#ifndef PEACEFUL
    // expected-note@-4{{Object 'a' is moved}}
    // expected-warning@-3{{Method called on moved-from object 'a'}}
    // expected-note@-4   {{Method called on moved-from object 'a'}}
#endif
  }
  {
    A a;
    A b;
    b = std::move(a);
    not_known_pass_by_rvalue_ref(std::move(a));
    a.foo(); // no-warning
  }
  {
    A a;
    A b;
    b = std::move(a);
    not_known_pass_by_ptr(&a);
    a.foo(); // no-warning
  }
  {
    A a;
    A b;
    b = std::move(a);
    not_known_pass_by_const_ptr(&a);
    a.foo();
#ifndef PEACEFUL
    // expected-note@-4{{Object 'a' is moved}}
    // expected-warning@-3{{Method called on moved-from object 'a'}}
    // expected-note@-4   {{Method called on moved-from object 'a'}}
#endif
  }
}

// A declaration statement containing multiple declarations sequences the
// initializer expressions.
void declarationSequenceTest() {
  {
    A a;
    A a1 = a, a2 = std::move(a); // no-warning
  }
  {
    A a;
    A a1 = std::move(a), a2 = a;
#ifndef PEACEFUL
    // expected-note@-2 {{Object 'a' is moved}}
    // expected-warning@-3 {{Moved-from object 'a' is copied}}
    // expected-note@-4    {{Moved-from object 'a' is copied}}
#endif
  }
}

// The logical operators && and || sequence their operands.
void logicalOperatorsSequenceTest() {
  {
    A a;
    if (a.foo() > 0 && A(std::move(a)).foo() > 0) {
#ifndef PEACEFUL
      // expected-note@-2 {{Assuming the condition is false}}
      // expected-note@-3 {{Left side of '&&' is false}}
      // expected-note@-4 {{Taking false branch}}
      // And the other report:
      // expected-note@-6 {{Assuming the condition is false}}
      // expected-note@-7 {{Left side of '&&' is false}}
      // expected-note@-8 {{Taking false branch}}
      A().bar();
#endif
    }
  }
  // A variation: Negate the result of the && (which pushes the && further down
  // into the AST).
  {
    A a;
    if (!(a.foo() > 0 && A(std::move(a)).foo() > 0)) {
#ifndef PEACEFUL
      // expected-note@-2 {{Assuming the condition is false}}
      // expected-note@-3 {{Left side of '&&' is false}}
      // expected-note@-4 {{Taking true branch}}
      // And the other report:
      // expected-note@-6 {{Assuming the condition is false}}
      // expected-note@-7 {{Left side of '&&' is false}}
      // expected-note@-8 {{Taking true branch}}
#endif
      A().bar();
    }
  }
  {
    A a;
    if (A(std::move(a)).foo() > 0 && a.foo() > 0) {
#ifndef PEACEFUL
      // expected-note@-2 {{Object 'a' is moved}}
      // expected-note@-3 {{Assuming the condition is true}}
      // expected-note@-4 {{Left side of '&&' is true}}
      // expected-warning@-5 {{Method called on moved-from object 'a'}}
      // expected-note@-6    {{Method called on moved-from object 'a'}}
      // And the other report:
      // expected-note@-8 {{Assuming the condition is false}}
      // expected-note@-9 {{Left side of '&&' is false}}
      // expected-note@-10{{Taking false branch}}
#endif
      A().bar();
    }
  }
  {
    A a;
    if (a.foo() > 0 || A(std::move(a)).foo() > 0) {
#ifndef PEACEFUL
      // expected-note@-2 {{Assuming the condition is true}}
      // expected-note@-3 {{Left side of '||' is true}}
      // expected-note@-4 {{Taking true branch}}
#endif
      A().bar();
    }
  }
  {
    A a;
    if (A(std::move(a)).foo() > 0 || a.foo() > 0) {
#ifndef PEACEFUL
      // expected-note@-2 {{Object 'a' is moved}}
      // expected-note@-3 {{Assuming the condition is false}}
      // expected-note@-4 {{Left side of '||' is false}}
      // expected-warning@-5 {{Method called on moved-from object 'a'}}
      // expected-note@-6    {{Method called on moved-from object 'a'}}
#endif
      A().bar();
    }
  }
}

// A range-based for sequences the loop variable declaration before the body.
void forRangeSequencesTest() {
  A v[2] = {A(), A()};
  for (A &a : v) {
    A b;
    b = std::move(a); // no-warning
  }
}

// If a variable is declared in an if statement, the declaration of the variable
// (which is treated like a reinitialization by the check) is sequenced before
// the evaluation of the condition (which constitutes a use).
void ifStmtSequencesDeclAndConditionTest() {
  for (int i = 0; i < 3; ++i) {
    if (A a = A()) {
      A b;
      b = std::move(a); // no-warning
    }
  }
}

struct C : public A {
  [[clang::reinitializes]] void reinit();
};

void subRegionMoveTest() {
  {
    A a;
    B b = std::move(a.b);
    a.b.foo();
#ifdef AGGRESSIVE
    // expected-note@-3{{Object 'b' is moved}}
    // expected-warning@-3{{Method called on moved-from object 'b'}}
    // expected-note@-4 {{Method called on moved-from object 'b'}}
#endif
  }
  {
    A a;
    A a1 = std::move(a);
    a.b.foo();
#ifdef AGGRESSIVE
    // expected-note@-3{{Calling move constructor for 'A'}}
    // expected-note@-4{{Returning from move constructor for 'A'}}
    // expected-warning@-4{{Method called on moved-from object 'b'}}
    // expected-note@-5{{Method called on moved-from object 'b'}}
#endif
  }
  // Don't report a misuse if any SuperRegion is already reported.
  {
    A a;
    A a1 = std::move(a);
    a.foo();
#ifndef PEACEFUL
    // expected-note@-3 {{Object 'a' is moved}}
    // expected-warning@-3 {{Method called on moved-from object 'a'}}
    // expected-note@-4    {{Method called on moved-from object 'a'}}
#endif
    a.b.foo();           // no-warning
  }
  {
    C c;
    C c1 = std::move(c);
    c.foo();
#ifndef PEACEFUL
    // expected-note@-3 {{Object 'c' is moved}}
    // expected-warning@-3 {{Method called on moved-from object 'c'}}
    // expected-note@-4    {{Method called on moved-from object 'c'}}
#endif
    c.b.foo();           // no-warning
  }
}

void resetSuperClass() {
  C c;
  C c1 = std::move(c);
  c.clear();
  C c2 = c; // no-warning
}

void resetSuperClass2() {
  C c;
  C c1 = std::move(c);
  c.reinit();
  C c2 = c; // no-warning
}

void reportSuperClass() {
  C c;
  C c1 = std::move(c);
  c.foo();
#ifndef PEACEFUL
  // expected-note@-3 {{Object 'c' is moved}}
  // expected-warning@-3 {{Method called on moved-from object 'c'}}
  // expected-note@-4    {{Method called on moved-from object 'c'}}
#endif
  C c2 = c;            // no-warning
}

struct Empty {};

Empty inlinedCall() {
  // Used to warn because region 'e' failed to be cleaned up because no symbols
  // have ever died during the analysis and the checkDeadSymbols callback
  // was skipped entirely.
  Empty e{};
  return e; // no-warning
}

void checkInlinedCallZombies() {
  while (true)
    inlinedCall();
}

void checkLoopZombies() {
  while (true) {
    Empty e{};
    Empty f = std::move(e); // no-warning
  }
}

void checkMoreLoopZombies1(bool flag) {
  while (flag) {
    Empty e{};
    if (true)
      e; // expected-warning {{expression result unused}}
    Empty f = std::move(e); // no-warning
  }
}

bool coin();

void checkMoreLoopZombies2(bool flag) {
  while (flag) {
    Empty e{};
    while (coin())
      e; // expected-warning {{expression result unused}}
    Empty f = std::move(e); // no-warning
  }
}

void checkMoreLoopZombies3(bool flag) {
  while (flag) {
    Empty e{};
    do
      e; // expected-warning {{expression result unused}}
    while (coin());
    Empty f = std::move(e); // no-warning
  }
}

void checkMoreLoopZombies4(bool flag) {
  while (flag) {
    Empty e{};
    for (; coin();)
      e; // expected-warning {{expression result unused}}
    Empty f = std::move(e); // no-warning
  }
}

struct MoveOnlyWithDestructor {
  MoveOnlyWithDestructor();
  ~MoveOnlyWithDestructor();
  MoveOnlyWithDestructor(const MoveOnlyWithDestructor &m) = delete;
  MoveOnlyWithDestructor(MoveOnlyWithDestructor &&m);
};

MoveOnlyWithDestructor foo() {
  MoveOnlyWithDestructor m;
  return m;
}

class HasSTLField {
  std::vector<int> V;
  void testVector() {
    // Warn even in non-aggressive mode when it comes to STL, because
    // in STL the object is left in "valid but unspecified state" after move.
    std::vector<int> W = std::move(V); // expected-note{{Object 'V' of type 'std::vector' is left in a valid but unspecified state after move}}
    V.push_back(123); // expected-warning{{Method called on moved-from object 'V'}}
                      // expected-note@-1{{Method called on moved-from object 'V'}}
  }

  std::unique_ptr<int> P;
  void testUniquePtr() {
    // unique_ptr remains in a well-defined state after move.
    std::unique_ptr<int> Q = std::move(P);
    P.get();
#ifdef AGGRESSIVE
    // expected-warning@-2{{Method called on moved-from object 'P'}}
    // expected-note@-4{{Object 'P' is moved}}
    // expected-note@-4{{Method called on moved-from object 'P'}}
#endif

    // Because that well-defined state is null, dereference is still UB.
    // Note that in aggressive mode we already warned about 'P',
    // so no extra warning is generated.
    *P += 1;
#ifndef AGGRESSIVE
    // expected-warning@-2{{Dereference of null smart pointer 'P' of type 'std::unique_ptr'}}
    // expected-note@-14{{Smart pointer 'P' of type 'std::unique_ptr' is reset to null when moved from}}
    // expected-note@-4{{Dereference of null smart pointer 'P' of type 'std::unique_ptr'}}
#endif

    // The program should have crashed by now.
    clang_analyzer_warnIfReached(); // no-warning
  }
};

void localRValueMove(A &&a) {
  A b = std::move(a);
  a.foo();
#ifndef PEACEFUL
  // expected-note@-3 {{Object 'a' is moved}}
  // expected-warning@-3 {{Method called on moved-from object 'a'}}
  // expected-note@-4    {{Method called on moved-from object 'a'}}
#endif
}

void localUniquePtr(std::unique_ptr<int> P) {
  // Even though unique_ptr is safe to use after move,
  // reusing a local variable this way usually indicates a bug.
  std::unique_ptr<int> Q = std::move(P);
  P.get();
#ifndef PEACEFUL
  // expected-note@-3 {{Object 'P' is moved}}
  // expected-warning@-3 {{Method called on moved-from object 'P'}}
  // expected-note@-4    {{Method called on moved-from object 'P'}}
#endif
}

void localUniquePtrWithArrow(std::unique_ptr<A> P) {
  std::unique_ptr<A> Q = std::move(P); // expected-note{{Smart pointer 'P' of type 'std::unique_ptr' is reset to null when moved from}}
  P->foo(); // expected-warning{{Dereference of null smart pointer 'P' of type 'std::unique_ptr'}}
            // expected-note@-1{{Dereference of null smart pointer 'P' of type 'std::unique_ptr'}}
}
