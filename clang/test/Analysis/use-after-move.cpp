// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.Move %s\
// RUN:  -std=c++11 -analyzer-output=text -analyzer-config eagerly-assume=false\
// RUN:  -analyzer-config exploration_strategy=unexplored_first_queue\
// RUN:  -analyzer-checker core,cplusplus.SmartPtr,debug.ExprInspection\
// RUN:  -verify=expected,peaceful,non-aggressive
// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.Move %s\
// RUN:  -std=c++11 -analyzer-output=text -analyzer-config eagerly-assume=false\
// RUN:  -analyzer-config exploration_strategy=dfs -DDFS\
// RUN:  -analyzer-checker core,cplusplus.SmartPtr,debug.ExprInspection\
// RUN:  -verify=expected,peaceful,non-aggressive
// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.Move %s\
// RUN:  -std=c++11 -analyzer-output=text -analyzer-config eagerly-assume=false\
// RUN:  -analyzer-config exploration_strategy=unexplored_first_queue\
// RUN:  -analyzer-config cplusplus.Move:WarnOn=KnownsOnly\
// RUN:  -analyzer-checker core,cplusplus.SmartPtr,debug.ExprInspection\
// RUN:  -verify=expected,non-aggressive
// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.Move -verify %s\
// RUN:  -std=c++11 -analyzer-output=text -analyzer-config eagerly-assume=false\
// RUN:  -analyzer-config exploration_strategy=dfs -DDFS\
// RUN:  -analyzer-config cplusplus.Move:WarnOn=KnownsOnly\
// RUN:  -analyzer-checker core,cplusplus.SmartPtr,debug.ExprInspection\
// RUN:  -verify=expected,non-aggressive
// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.Move %s\
// RUN:  -std=c++11 -analyzer-output=text -analyzer-config eagerly-assume=false\
// RUN:  -analyzer-config exploration_strategy=unexplored_first_queue\
// RUN:  -analyzer-config cplusplus.Move:WarnOn=All\
// RUN:  -analyzer-checker core,cplusplus.SmartPtr,debug.ExprInspection\
// RUN:  -verify=expected,peaceful,aggressive
// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.Move %s\
// RUN:  -std=c++11 -analyzer-output=text -analyzer-config eagerly-assume=false\
// RUN:  -analyzer-config exploration_strategy=dfs -DDFS\
// RUN:  -analyzer-config cplusplus.Move:WarnOn=All\
// RUN:  -analyzer-checker core,cplusplus.SmartPtr,debug.ExprInspection\
// RUN:  -verify=expected,peaceful,aggressive

// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.Move \
// RUN:   -analyzer-config cplusplus.Move:WarnOn="a bunch of things" \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-MOVE-INVALID-VALUE

// CHECK-MOVE-INVALID-VALUE: (frontend): invalid input for checker option
// CHECK-MOVE-INVALID-VALUE-SAME: 'cplusplus.Move:WarnOn', that expects either
// CHECK-MOVE-INVALID-VALUE-SAME: "KnownsOnly", "KnownsAndLocals" or "All"
// CHECK-MOVE-INVALID-VALUE-SAME: string value

// Tests checker-messages printing.
// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.Move %s\
// RUN:  -std=c++11 -analyzer-output=text -analyzer-config eagerly-assume=false\
// RUN:  -analyzer-config exploration_strategy=dfs -DDFS\
// RUN:  -analyzer-config cplusplus.Move:WarnOn=All -DAGGRESSIVE_DFS\
// RUN:  -analyzer-checker core,cplusplus.SmartPtr,debug.ExprInspection\
// RUN:  -verify=expected,peaceful,aggressive %s 2>&1 | FileCheck %s

#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_warnIfReached();
void clang_analyzer_printState();

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
  A(A &&other) : i(other.i), d(other.d), b(std::move(other.b)) { // aggressive-note{{Object 'b' is moved}}
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
    A b = std::move(a); // peaceful-note {{Object 'a' is moved}}

#ifdef AGGRESSIVE_DFS
    clang_analyzer_printState();

// CHECK:      "checker_messages": [
// CHECK-NEXT:   { "checker": "cplusplus.Move", "messages": [
// CHECK-NEXT:     "Moved-from objects :",
// CHECK:          "a: moved",
// CHECK:          ""
// CHECK-NEXT:   ]}
// CHECK-NEXT: ]
#endif

    a.foo(); // peaceful-warning {{Method called on moved-from object 'a'}}
             // peaceful-note@-1 {{Method called on moved-from object 'a'}}
  }
  {
    A a;
    A b = std::move(a); // peaceful-note {{Object 'a' is moved}}
    b = a; // peaceful-warning {{Moved-from object 'a' is copied}}
           // peaceful-note@-1 {{Moved-from object 'a' is copied}}
  }
  {
    A a;
    A b = std::move(a); // peaceful-note {{Object 'a' is moved}}
    b = std::move(a); // peaceful-warning {{Moved-from object 'a' is moved}}
                      // peaceful-note@-1 {{Moved-from object 'a' is moved}}
  }
}

void simpleMoveAssignementTest() {
  {
    A a;
    A b;
    b = std::move(a); // peaceful-note {{Object 'a' is moved}}
    a.foo(); // peaceful-warning {{Method called on moved-from object 'a'}}
             // peaceful-note@-1 {{Method called on moved-from object 'a'}}
  }
  {
    A a;
    A b;
    b = std::move(a); // peaceful-note {{Object 'a' is moved}}
    A c(a); // peaceful-warning {{Moved-from object 'a' is copied}}
            // peaceful-note@-1 {{Moved-from object 'a' is copied}}
  }
  {
    A a;
    A b;
    b = std::move(a); // peaceful-note {{Object 'a' is moved}}
    A c(std::move(a)); // peaceful-warning {{Moved-from object 'a' is moved}}
                       // peaceful-note@-1 {{Moved-from object 'a' is moved}}
  }
}

void moveInInitListTest() {
  struct S {
    A a;
  };
  A a;
  S s{std::move(a)}; // peaceful-note {{Object 'a' is moved}}
  a.foo(); // peaceful-warning {{Method called on moved-from object 'a'}}
           // peaceful-note@-1 {{Method called on moved-from object 'a'}}
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
    if (i == 1) { // peaceful-note 2 {{Assuming 'i' is not equal to 1}}
                  // peaceful-note@-1 2 {{Taking false branch}}
      A b;
      b = std::move(a);
      a = A();
    }
    if (i == 2) { // peaceful-note 2 {{Assuming 'i' is not equal to 2}}
                  // peaceful-note@-1 2 {{Taking false branch}}
      a.foo();    // no-warning
    }
  }
  {
    A a;
    if (i == 1) { // peaceful-note 2 {{'i' is not equal to 1}}
                  // peaceful-note@-1 2 {{Taking false branch}}
      std::move(a);
    }
    if (i == 2) { // peaceful-note 2 {{'i' is not equal to 2}}
                  // peaceful-note@-1 2 {{Taking false branch}}
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
    b = std::move(a); // peaceful-note {{Object 'a' is moved}}
    a.foo(); // peaceful-warning {{Method called on moved-from object 'a'}}
             // peaceful-note@-1 {{Method called on moved-from object 'a'}}
  }
  // If a path exist where we not reinitialize the variable we report a bug.
  {
    A a;
    A b;
    b = std::move(a); // peaceful-note {{Object 'a' is moved}}
    if (i < 10) { // peaceful-note {{Assuming 'i' is >= 10}}
                  // peaceful-note@-1 {{Taking false branch}}
      a = A();
    }
    if (i > 5) { // peaceful-note {{'i' is > 5}}
                 // peaceful-note@-1 {{Taking true branch}}
      a.foo(); // peaceful-warning {{Method called on moved-from object 'a'}}
               // peaceful-note@-1 {{Method called on moved-from object 'a'}}
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
    // FIXME: Execution doesn't jump to the end of the function yet.
    for (int i = 0; i < bignum(); i++) { // peaceful-note {{Loop condition is false. Execution jumps to the end of the function}}
      rightRefCall(std::move(a));        // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < 2; i++) { // peaceful-note    {{Loop condition is true.  Entering loop body}}
                                  // peaceful-note@-1 {{Loop condition is true.  Entering loop body}}
                                  // peaceful-note@-2 {{Loop condition is false. Execution jumps to the end of the function}}
      rightRefCall(std::move(a)); // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < bignum(); i++) { // peaceful-note {{Loop condition is false. Execution jumps to the end of the function}}
      leftRefCall(a);                    // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < 2; i++) { // peaceful-note    {{Loop condition is true.  Entering loop body}}
                                  // peaceful-note@-1 {{Loop condition is true.  Entering loop body}}
                                  // peaceful-note@-2 {{Loop condition is false. Execution jumps to the end of the function}}
      leftRefCall(a);             // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < bignum(); i++) { // peaceful-note {{Loop condition is false. Execution jumps to the end of the function}}
      constCopyOrMoveCall(a);            // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < 2; i++) { // peaceful-note    {{Loop condition is true.  Entering loop body}}
                                  // peaceful-note@-1 {{Loop condition is true.  Entering loop body}}
                                  // peaceful-note@-2 {{Loop condition is false. Execution jumps to the end of the function}}
      constCopyOrMoveCall(a);     // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < bignum(); i++) { // peaceful-note {{Loop condition is false. Execution jumps to the end of the function}}
      moveInsideFunctionCall(a);         // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < 2; i++) { // peaceful-note    {{Loop condition is true.  Entering loop body}}
                                  // peaceful-note@-1 {{Loop condition is true.  Entering loop body}}
                                  // peaceful-note@-2 {{Loop condition is false. Execution jumps to the end of the function}}
      moveInsideFunctionCall(a);  // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < bignum(); i++) { // peaceful-note {{Loop condition is false. Execution jumps to the end of the function}}
      copyOrMoveCall(a);                 // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < 2; i++) { // peaceful-note    {{Loop condition is true.  Entering loop body}}
                                  // peaceful-note@-1 {{Loop condition is true.  Entering loop body}}
                                  // peaceful-note@-2 {{Loop condition is false. Execution jumps to the end of the function}}
      copyOrMoveCall(a);          // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < bignum(); i++) { // peaceful-note    {{Loop condition is true.  Entering loop body}}
                                         // peaceful-note@-1 {{Loop condition is true.  Entering loop body}}
      constCopyOrMoveCall(std::move(a)); // peaceful-note {{Object 'a' is moved}}
                                         // peaceful-warning@-1 {{Moved-from object 'a' is moved}}
                                         // peaceful-note@-2    {{Moved-from object 'a' is moved}}
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
  b = std::move(a); // peaceful-note {{Object 'a' is moved}}

  if (cond) { // peaceful-note {{Assuming 'cond' is true}}
              // peaceful-note@-1 {{Taking true branch}}
    a.foo(); // peaceful-warning {{Method called on moved-from object 'a'}}
             // peaceful-note@-1 {{Method called on moved-from object 'a'}}
  }
  if (cond) {
    a.bar(); // no-warning
  }

  a.bar(); // no-warning
}

void uniqueTest2() {
  A a;
  A a1 = std::move(a); // peaceful-note {{Object 'a' is moved}}
  a.foo(); // peaceful-warning {{Method called on moved-from object 'a'}}
           // peaceful-note@-1    {{Method called on moved-from object 'a'}}

  A a2 = std::move(a); // no-warning
  a.foo();             // no-warning
}

// There are exceptions where we assume in general that the method works fine
//even on moved-from objects.
void moveSafeFunctionsTest() {
  A a;
  A b = std::move(a); // peaceful-note {{Object 'a' is moved}}
  a.empty();          // no-warning
  a.isEmpty();        // no-warning
  (void)a;            // no-warning
  (bool)a;            // expected-warning {{expression result unused}}
  a.foo(); // peaceful-warning {{Method called on moved-from object 'a'}}
           // peaceful-note@-1 {{Method called on moved-from object 'a'}}
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
    b = std::move(a); // aggressive-note {{Object 'a' is moved}}

    a.foo(); // aggressive-warning {{Method called on moved-from object 'a'}}
             // aggressive-note@-1 {{Method called on moved-from object 'a'}}

    b = std::move(static_a); // aggressive-note {{Object 'static_a' is moved}}
    static_a.foo(); // aggressive-warning {{Method called on moved-from object 'static_a'}}
                    // aggressive-note@-1 {{Method called on moved-from object 'static_a'}}
  }
};

void PtrAndArrayTest() {
  A *Ptr = new A(1, 1.5);
  A Arr[10];
  Arr[2] = std::move(*Ptr); // aggressive-note{{Object is moved}}
  (*Ptr).foo(); // aggressive-warning{{Method called on moved-from object}}
                // aggressive-note@-1{{Method called on moved-from object}}

  Ptr = &Arr[1];
  Arr[3] = std::move(Arr[1]); // aggressive-note {{Object is moved}}
  Ptr->foo(); // aggressive-warning {{Method called on moved-from object}}
              // aggressive-note@-1 {{Method called on moved-from object}}

  Arr[3] = std::move(Arr[2]); // aggressive-note{{Object is moved}}
  Arr[2].foo(); // aggressive-warning{{Method called on moved-from object}}
                // aggressive-note@-1{{Method called on moved-from object}}

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
    if (i > 0) { // peaceful-note {{Assuming 'i' is > 0}}
                 // peaceful-note@-1 {{Taking true branch}}
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
    // peaceful-note@-1 {{'i' is > 0}}
    // peaceful-note@-2 {{'?' condition is true}}
  }
  // A variation on the theme above.
  {
    A a;
    a.foo() > 0 ? a.foo() : A(std::move(a)).foo();
#ifdef DFS
    // peaceful-note@-2 {{Assuming the condition is false}}
    // peaceful-note@-3 {{'?' condition is false}}
#else
    // peaceful-note@-5 {{Assuming the condition is true}}
    // peaceful-note@-6 {{'?' condition is true}}
#endif
  }
  // Same thing, but with a switch statement.
  {
    A a, b;
    switch (i) { // peaceful-note {{Control jumps to 'case 1:'}}
    case 1:
      b = std::move(a); // no-warning
      // FIXME: Execution doesn't jump to the end of the function yet.
      break; // peaceful-note {{Execution jumps to the end of the function}}
    case 2:
      a.foo(); // no-warning
      break;
    }
  }
  // However, if there's a fallthrough, we do warn.
  {
    A a, b;
    switch (i) { // peaceful-note {{Control jumps to 'case 1:'}}
    case 1:
      b = std::move(a); // peaceful-note {{Object 'a' is moved}}
    case 2:
      a.foo(); // peaceful-warning {{Method called on moved-from object 'a'}}
               // peaceful-note@-1 {{Method called on moved-from object 'a'}}
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
  a.bar(); // peaceful-warning {{Method called on moved-from object 'a'}}
           // peaceful-note@-1 {{Method called on moved-from object 'a'}}
}

void interFunTest2() {
  A a;
  A b;
  b = std::move(a); // peaceful-note {{Object 'a' is moved}}
  interFunTest1(a); // peaceful-note {{Calling 'interFunTest1'}}
}

void foobar(A a, int i);
void foobar(int i, A a);

void paramEvaluateOrderTest() {
  A a;
  foobar(std::move(a), a.getI()); // peaceful-note {{Object 'a' is moved}}
                                  // peaceful-warning@-1 {{Method called on moved-from object 'a'}}
                                  // peaceful-note@-2    {{Method called on moved-from object 'a'}}

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
    b = std::move(a); // peaceful-note{{Object 'a' is moved}}
    not_known_pass_by_const_ref(a);
    a.foo(); // peaceful-warning {{Method called on moved-from object 'a'}}
             // peaceful-note@-1 {{Method called on moved-from object 'a'}}
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
    b = std::move(a); // peaceful-note {{Object 'a' is moved}}
    not_known_pass_by_const_ptr(&a);
    a.foo(); // peaceful-warning {{Method called on moved-from object 'a'}}
             // peaceful-note@-1 {{Method called on moved-from object 'a'}}
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
    A a1 = std::move(a), a2 = a; // peaceful-note {{Object 'a' is moved}}
                                 // peaceful-warning@-1 {{Moved-from object 'a' is copied}}
                                 // peaceful-note@-2    {{Moved-from object 'a' is copied}}
  }
}

// The logical operators && and || sequence their operands.
void logicalOperatorsSequenceTest() {
  {
    A a;
    if (a.foo() > 0 && A(std::move(a)).foo() > 0) { // peaceful-note    {{Assuming the condition is false}}
                                                    // peaceful-note@-1 {{Left side of '&&' is false}}
                                                    // peaceful-note@-2 {{Taking false branch}}
                                                    // And the other report:
                                                    // peaceful-note@-4 {{Assuming the condition is false}}
                                                    // peaceful-note@-5 {{Left side of '&&' is false}}
                                                    // peaceful-note@-6 {{Taking false branch}}
      A().bar();
    }
  }
  // A variation: Negate the result of the && (which pushes the && further down
  // into the AST).
  {
    A a;
    if (!(a.foo() > 0 && A(std::move(a)).foo() > 0)) { // peaceful-note    {{Assuming the condition is false}}
                                                       // peaceful-note@-1 {{Left side of '&&' is false}}
                                                       // peaceful-note@-2 {{Taking true branch}}
                                                       // And the other report:
                                                       // peaceful-note@-4 {{Assuming the condition is false}}
                                                       // peaceful-note@-5 {{Left side of '&&' is false}}
                                                       // peaceful-note@-6 {{Taking true branch}}
      A().bar();
    }
  }
  {
    A a;
    if (A(std::move(a)).foo() > 0 && a.foo() > 0) { // peaceful-note    {{Object 'a' is moved}}
                                                    // peaceful-note@-1 {{Assuming the condition is true}}
                                                    // peaceful-note@-2 {{Left side of '&&' is true}}
                                                    // peaceful-warning@-3 {{Method called on moved-from object 'a'}}
                                                    // peaceful-note@-4    {{Method called on moved-from object 'a'}}
                                                    // And the other report:
                                                    // peaceful-note@-6 {{Assuming the condition is false}}
                                                    // peaceful-note@-7 {{Left side of '&&' is false}}
                                                    // peaceful-note@-8 {{Taking false branch}}
      A().bar();
    }
  }
  {
    A a;
    if (a.foo() > 0 || A(std::move(a)).foo() > 0) { // peaceful-note    {{Assuming the condition is true}}
                                                    // peaceful-note@-1 {{Left side of '||' is true}}
                                                    // peaceful-note@-2 {{Taking true branch}}
      A().bar();
    }
  }
  {
    A a;
    if (A(std::move(a)).foo() > 0 || a.foo() > 0) { // peaceful-note {{Object 'a' is moved}}
                                                    // peaceful-note@-1 {{Assuming the condition is false}}
                                                    // peaceful-note@-2 {{Left side of '||' is false}}
                                                    // peaceful-warning@-3 {{Method called on moved-from object 'a'}}
                                                    // peaceful-note@-4    {{Method called on moved-from object 'a'}}
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
    B b = std::move(a.b); // aggressive-note {{Object 'b' is moved}}
    a.b.foo(); // aggressive-warning {{Method called on moved-from object 'b'}}
               // aggressive-note@-1 {{Method called on moved-from object 'b'}}
  }
  {
    A a;
    A a1 = std::move(a); // aggressive-note {{Calling move constructor for 'A'}}
                         // aggressive-note@-1 {{Returning from move constructor for 'A'}}
    a.b.foo(); // aggressive-warning{{Method called on moved-from object 'b'}}
               // aggressive-note@-1{{Method called on moved-from object 'b'}}
  }
  // Don't report a misuse if any SuperRegion is already reported.
  {
    A a;
    A a1 = std::move(a); // peaceful-note {{Object 'a' is moved}}
    a.foo(); // peaceful-warning {{Method called on moved-from object 'a'}}
             // peaceful-note@-1 {{Method called on moved-from object 'a'}}
    a.b.foo(); // no-warning
  }
  {
    C c;
    C c1 = std::move(c); // peaceful-note {{Object 'c' is moved}}
    c.foo(); // peaceful-warning {{Method called on moved-from object 'c'}}
             // peaceful-note@-1 {{Method called on moved-from object 'c'}}
    c.b.foo(); // no-warning
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
  C c1 = std::move(c); // peaceful-note {{Object 'c' is moved}}
  c.foo(); // peaceful-warning {{Method called on moved-from object 'c'}}
           // peaceful-note@-1 {{Method called on moved-from object 'c'}}
  C c2 = c; // no-warning
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
    std::vector<int> W = std::move(V); // expected-note {{Object 'V' of type 'std::vector' is left in a valid but unspecified state after move}}
    V.push_back(123); // expected-warning {{Method called on moved-from object 'V'}}
                      // expected-note@-1 {{Method called on moved-from object 'V'}}
  }

  std::unique_ptr<int> P;
  void testUniquePtr() {
    // unique_ptr remains in a well-defined state after move.
    std::unique_ptr<int> Q = std::move(P); // aggressive-note {{Object 'P' is moved}}
                                           // non-aggressive-note@-1 {{Smart pointer 'P' of type 'std::unique_ptr' is reset to null when moved from}}
    P.get(); // aggressive-warning{{Method called on moved-from object 'P'}}
             // aggressive-note@-1{{Method called on moved-from object 'P'}}

    // Because that well-defined state is null, dereference is still UB.
    // Note that in aggressive mode we already warned about 'P',
    // so no extra warning is generated.
    *P += 1; // non-aggressive-warning{{Dereference of null smart pointer 'P' of type 'std::unique_ptr'}}
             // non-aggressive-note@-1{{Dereference of null smart pointer 'P' of type 'std::unique_ptr'}}

    // The program should have crashed by now.
    clang_analyzer_warnIfReached(); // no-warning
  }
};

void localRValueMove(A &&a) {
  A b = std::move(a); // peaceful-note {{Object 'a' is moved}}
  a.foo(); // peaceful-warning {{Method called on moved-from object 'a'}}
           // peaceful-note@-1 {{Method called on moved-from object 'a'}}
}

void localUniquePtr(std::unique_ptr<int> P) {
  // Even though unique_ptr is safe to use after move,
  // reusing a local variable this way usually indicates a bug.
  std::unique_ptr<int> Q = std::move(P); // peaceful-note {{Object 'P' is moved}}
  P.get(); // peaceful-warning {{Method called on moved-from object 'P'}}
           // peaceful-note@-1 {{Method called on moved-from object 'P'}}
}

void localUniquePtrWithArrow(std::unique_ptr<A> P) {
  std::unique_ptr<A> Q = std::move(P); // expected-note{{Smart pointer 'P' of type 'std::unique_ptr' is reset to null when moved from}}
  P->foo(); // expected-warning{{Dereference of null smart pointer 'P' of type 'std::unique_ptr'}}
            // expected-note@-1{{Dereference of null smart pointer 'P' of type 'std::unique_ptr'}}
}

void getAfterMove(std::unique_ptr<A> P) {
  std::unique_ptr<A> Q = std::move(P); // peaceful-note {{Object 'P' is moved}}

  // TODO: Explain why (bool)P is false.
  if (P) // peaceful-note{{Taking false branch}}
    clang_analyzer_warnIfReached(); // no-warning

  A *a = P.get(); // peaceful-warning {{Method called on moved-from object 'P'}}
                  // peaceful-note@-1 {{Method called on moved-from object 'P'}}

  // TODO: Warn on a null dereference here.
  a->foo();
}
