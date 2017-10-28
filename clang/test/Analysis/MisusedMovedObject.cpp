// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.cplusplus.MisusedMovedObject -std=c++11 -verify -analyzer-output=text %s

namespace std {

template <typename>
struct remove_reference;

template <typename _Tp>
struct remove_reference { typedef _Tp type; };

template <typename _Tp>
struct remove_reference<_Tp &> { typedef _Tp type; };

template <typename _Tp>
struct remove_reference<_Tp &&> { typedef _Tp type; };

template <typename _Tp>
typename remove_reference<_Tp>::type &&move(_Tp &&__t) {
  return static_cast<typename remove_reference<_Tp>::type &&>(__t);
}

template <typename _Tp>
_Tp &&forward(typename remove_reference<_Tp>::type &__t) noexcept {
  return static_cast<_Tp &&>(__t);
}

template <class T>
void swap(T &a, T &b) {
  T c(std::move(a));
  a = std::move(b);
  b = std::move(c);
}

} // namespace std

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
  A(A &&other) : i(other.i), d(other.d), b(std::move(other.b)) { // expected-note {{'b' became 'moved-from' here}}
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
  bool empty() const;
  bool isEmpty() const;
  operator bool() const;
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
    A b = std::move(a); // expected-note {{'a' became 'moved-from' here}}
    a.foo();            // expected-warning {{Method call on a 'moved-from' object 'a'}} expected-note {{Method call on a 'moved-from' object 'a'}}
  }
  {
    A a;
    A b = std::move(a); // expected-note {{'a' became 'moved-from' here}}
    b = a;              // expected-warning {{Copying a 'moved-from' object 'a'}} expected-note {{Copying a 'moved-from' object 'a'}}
  }
  {
    A a;
    A b = std::move(a); // expected-note {{'a' became 'moved-from' here}}
    b = std::move(a);   // expected-warning {{Moving a 'moved-from' object 'a'}} expected-note {{Moving a 'moved-from' object 'a'}}
  }
}

void simpleMoveAssignementTest() {
  {
    A a;
    A b;
    b = std::move(a); // expected-note {{'a' became 'moved-from' here}}
    a.foo();          // expected-warning {{Method call on a 'moved-from' object 'a'}} expected-note {{Method call on a 'moved-from' object 'a'}}
  }
  {
    A a;
    A b;
    b = std::move(a); // expected-note {{'a' became 'moved-from' here}}
    A c(a);           // expected-warning {{Copying a 'moved-from' object 'a'}} expected-note {{Copying a 'moved-from' object 'a'}}
  }
  {
    A a;
    A b;
    b = std::move(a);  // expected-note {{'a' became 'moved-from' here}}
    A c(std::move(a)); // expected-warning {{Moving a 'moved-from' object 'a'}} expected-note {{Moving a 'moved-from' object 'a'}}
  }
}

void moveInInitListTest() {
  struct S {
    A a;
  };
  A a;
  S s{std::move(a)}; // expected-note {{'a' became 'moved-from' here}}
  a.foo();           // expected-warning {{Method call on a 'moved-from' object 'a'}} expected-note {{Method call on a 'moved-from' object 'a'}}
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
    if (i == 1) { // expected-note {{Assuming 'i' is not equal to 1}} expected-note {{Taking false branch}}
      // expected-note@-1 {{Assuming 'i' is not equal to 1}} expected-note@-1 {{Taking false branch}}
      A b;
      b = std::move(a);
      a = A();
    }
    if (i == 2) { // expected-note {{Assuming 'i' is not equal to 2}} expected-note {{Taking false branch}}
      //expected-note@-1 {{Assuming 'i' is not equal to 2}} expected-note@-1 {{Taking false branch}}
      a.foo();    // no-warning
    }
  }
  {
    A a;
    if (i == 1) { // expected-note {{Taking false branch}} expected-note {{Taking false branch}}
      std::move(a);
    }
    if (i == 2) { // expected-note {{Taking false branch}} expected-note {{Taking false branch}}
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
    b = std::move(a); // expected-note {{'a' became 'moved-from' here}}
    a.foo();          // expected-warning {{Method call on a 'moved-from' object 'a'}} expected-note {{Method call on a 'moved-from' object 'a'}}
  }
  // If a path exist where we not reinitialize the variable we report a bug.
  {
    A a;
    A b;
    b = std::move(a); // expected-note {{'a' became 'moved-from' here}}
    if (i < 10) {     // expected-note {{Assuming 'i' is >= 10}} expected-note {{Taking false branch}}
      a = A();
    }
    if (i > 5) { // expected-note {{Taking true branch}}
      a.foo();   // expected-warning {{Method call on a 'moved-from' object 'a'}} expected-note {{Method call on a 'moved-from' object 'a'}}
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
    for (int i = 0; i < bignum(); i++) { // expected-note {{Loop condition is false. Execution jumps to the end of the function}}
      rightRefCall(std::move(a));        // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < 2; i++) { // expected-note {{Loop condition is true.  Entering loop body}}
      //expected-note@-1 {{Loop condition is true.  Entering loop body}}
			//expected-note@-2 {{Loop condition is false. Execution jumps to the end of the function}}
      rightRefCall(std::move(a)); // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < bignum(); i++) { // expected-note {{Loop condition is false. Execution jumps to the end of the function}}
      leftRefCall(a);                    // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < 2; i++) { // expected-note {{Loop condition is true.  Entering loop body}} 
      //expected-note@-1 {{Loop condition is true.  Entering loop body}}
			//expected-note@-2 {{Loop condition is false. Execution jumps to the end of the function}}
      leftRefCall(a);             // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < bignum(); i++) { // expected-note {{Loop condition is false. Execution jumps to the end of the function}}
      constCopyOrMoveCall(a);            // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < 2; i++) { // expected-note {{Loop condition is true.  Entering loop body}} 
      //expected-note@-1 {{Loop condition is true.  Entering loop body}}
			//expected-note@-2 {{Loop condition is false. Execution jumps to the end of the function}}
      constCopyOrMoveCall(a);     // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < bignum(); i++) { // expected-note {{Loop condition is false. Execution jumps to the end of the function}}
      moveInsideFunctionCall(a);         // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < 2; i++) { // expected-note {{Loop condition is true.  Entering loop body}}
      //expected-note@-1 {{Loop condition is true.  Entering loop body}}
			//expected-note@-2 {{Loop condition is false. Execution jumps to the end of the function}}
      moveInsideFunctionCall(a);  // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < bignum(); i++) { // expected-note {{Loop condition is false. Execution jumps to the end of the function}}
      copyOrMoveCall(a);                 // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < 2; i++) { // expected-note {{Loop condition is true.}}
      //expected-note@-1 {{Loop condition is true.  Entering loop body}}
			//expected-note@-2 {{Loop condition is false. Execution jumps to the end of the function}}
      copyOrMoveCall(a);          // no-warning
    }
  }
  {
    A a;
    for (int i = 0; i < bignum(); i++) { // expected-note {{Loop condition is true.  Entering loop body}} expected-note {{Loop condition is true.  Entering loop body}}
      constCopyOrMoveCall(std::move(a)); // expected-warning {{Moving a 'moved-from' object 'a'}} expected-note {{Moving a 'moved-from' object 'a'}}
      // expected-note@-1 {{'a' became 'moved-from' here}}
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

//report a usage of a moved-from object only at the first use
void uniqueTest(bool cond) {
  A a(42, 42.0);
  A b;
  b = std::move(a); // expected-note {{'a' became 'moved-from' here}}

  if (cond) { // expected-note {{Assuming 'cond' is not equal to 0}} expected-note {{Taking true branch}}
    a.foo();  // expected-warning {{Method call on a 'moved-from' object 'a'}} expected-note {{Method call on a 'moved-from' object 'a'}}
  }
  if (cond) {
    a.bar(); // no-warning
  }

  a.bar(); // no-warning
}

void uniqueTest2() {
  A a;
  A a1 = std::move(a); // expected-note {{'a' became 'moved-from' here}}
  a.foo();             // expected-warning {{Method call on a 'moved-from' object 'a'}} expected-note {{Method call on a 'moved-from' object 'a'}}

  A a2 = std::move(a); // no-warning
  a.foo();             // no-warning
}

// There are exceptions where we assume in general that the method works fine
//even on moved-from objects.
void moveSafeFunctionsTest() {
  A a;
  A b = std::move(a); // expected-note {{'a' became 'moved-from' here}}
  a.empty();          // no-warning
  a.isEmpty();        // no-warning
  (void)a;            // no-warning
  (bool)a;            // expected-warning {{expression result unused}}
  a.foo();            // expected-warning {{Method call on a 'moved-from' object 'a'}} expected-note {{Method call on a 'moved-from' object 'a'}}
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
    b = std::move(a); // expected-note {{'a' became 'moved-from' here}}
    a.foo();          // expected-warning {{Method call on a 'moved-from' object}} expected-note {{Method call on a 'moved-from' object 'a'}}

    b = std::move(static_a); // expected-note {{'static_a' became 'moved-from' here}}
    static_a.foo();          // expected-warning {{Method call on a 'moved-from' object 'static_a'}} expected-note {{Method call on a 'moved-from' object 'static_a'}}
  }
};

void PtrAndArrayTest() {
  A *Ptr = new A(1, 1.5);
  A Arr[10];
  Arr[2] = std::move(*Ptr); // expected-note {{Became 'moved-from' here}}
  (*Ptr).foo();             // expected-warning {{Method call on a 'moved-from' object}} expected-note {{Method call on a 'moved-from' object}}

  Ptr = &Arr[1];
  Arr[3] = std::move(Arr[1]); // expected-note {{Became 'moved-from' here}}
  Ptr->foo();                 // expected-warning {{Method call on a 'moved-from' object}} expected-note {{Method call on a 'moved-from' object}}

  Arr[3] = std::move(Arr[2]); // expected-note {{Became 'moved-from' here}}
  Arr[2].foo();               // expected-warning {{Method call on a 'moved-from' object}} expected-note {{Method call on a 'moved-from' object}}

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
    if (i > 0) { // expected-note {{Assuming 'i' is > 0}} expected-note {{Taking true branch}}
      A b;
      b = std::move(a);
    } else {
      a.foo(); // no-warning
    }
  }
  // Same thing, but with a ternary operator.
  {
    A a, b;
    i > 0 ? (void)(b = std::move(a)) : a.bar(); // no-warning  // expected-note {{'?' condition is true}}
  }
  // A variation on the theme above.
  {
    A a;
    a.foo() > 0 ? a.foo() : A(std::move(a)).foo(); // expected-note {{Assuming the condition is false}} expected-note {{'?' condition is false}}
  }
  // Same thing, but with a switch statement.
  {
    A a, b;
    switch (i) { // expected-note {{Control jumps to 'case 1:'  at line 483}}
    case 1:
      b = std::move(a); // no-warning
      break;            // expected-note {{Execution jumps to the end of the function}}
    case 2:
      a.foo(); // no-warning
      break;
    }
  }
  // However, if there's a fallthrough, we do warn.
  {
    A a, b;
    switch (i) { // expected-note {{Control jumps to 'case 1:'  at line 495}}
    case 1:
      b = std::move(a); // expected-note {{'a' became 'moved-from' here}}
    case 2:
      a.foo(); // expected-warning {{Method call on a 'moved-from' object}} expected-note {{Method call on a 'moved-from' object 'a'}}
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
  a.bar(); // expected-warning {{Method call on a 'moved-from' object 'a'}} expected-note {{Method call on a 'moved-from' object 'a'}}
}

void interFunTest2() {
  A a;
  A b;
  b = std::move(a); // expected-note {{'a' became 'moved-from' here}}
  interFunTest1(a); // expected-note {{Calling 'interFunTest1'}}
}

void foobar(A a, int i);
void foobar(int i, A a);

void paramEvaluateOrderTest() {
  A a;
  foobar(std::move(a), a.getI()); // expected-warning {{Method call on a 'moved-from' object 'a'}} expected-note {{Method call on a 'moved-from' object 'a'}}
  // expected-note@-1 {{'a' became 'moved-from' here}}

  //FALSE NEGATIVE since parameters evaluate order is undefined
  foobar(a.getI(), std::move(a)); //no-warning
}

void not_known(A &a);
void not_known(A *a);

void regionAndPointerEscapeTest() {
  {
    A a;
    A b;
    b = std::move(a);
    not_known(a);
    a.foo(); //no-warning
  }
  {
    A a;
    A b;
    b = std::move(a);
    not_known(&a);
    a.foo(); // no-warning
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
    A a1 = std::move(a), a2 = a; // expected-warning {{Copying a 'moved-from' object 'a'}} expected-note {{Copying a 'moved-from' object 'a'}}
    // expected-note@-1 {{'a' became 'moved-from' here}}
  }
}

// The logical operators && and || sequence their operands.
void logicalOperatorsSequenceTest() {
  {
    A a;
    if (a.foo() > 0 && A(std::move(a)).foo() > 0) { // expected-note {{Assuming the condition is false}} expected-note {{Assuming the condition is false}} 
      // expected-note@-1 {{Left side of '&&' is false}} expected-note@-1 {{Left side of '&&' is false}}
			//expected-note@-2 {{Taking false branch}} expected-note@-2 {{Taking false branch}}
      A().bar();
    }
  }
  // A variation: Negate the result of the && (which pushes the && further down
  // into the AST).
  {
    A a;
    if (!(a.foo() > 0 && A(std::move(a)).foo() > 0)) { // expected-note {{Assuming the condition is false}} expected-note {{Assuming the condition is false}}
      // expected-note@-1 {{Left side of '&&' is false}} expected-note@-1 {{Left side of '&&' is false}}
      // expected-note@-2 {{Taking true branch}} expected-note@-2 {{Taking true branch}}
      A().bar();
    }
  }
  {
    A a;
    if (A(std::move(a)).foo() > 0 && a.foo() > 0) { // expected-warning {{Method call on a 'moved-from' object 'a'}} expected-note {{Method call on a 'moved-from' object 'a'}}
      // expected-note@-1 {{'a' became 'moved-from' here}} expected-note@-1 {{Assuming the condition is true}} expected-note@-1 {{Assuming the condition is false}}
      // expected-note@-2 {{Left side of '&&' is false}} expected-note@-2 {{Left side of '&&' is true}}
      // expected-note@-3 {{Taking false branch}}
      A().bar();
    }
  }
  {
    A a;
    if (a.foo() > 0 || A(std::move(a)).foo() > 0) { // expected-note {{Assuming the condition is true}} 
			//expected-note@-1 {{Left side of '||' is true}}
			//expected-note@-2 {{Taking true branch}}
      A().bar();
    }
  }
  {
    A a;
    if (A(std::move(a)).foo() > 0 || a.foo() > 0) { // expected-warning {{Method call on a 'moved-from' object 'a'}} expected-note {{Method call on a 'moved-from' object 'a'}}
      // expected-note@-1 {{'a' became 'moved-from' here}} expected-note@-1 {{Assuming the condition is false}} expected-note@-1 {{Left side of '||' is false}}
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

class C : public A {};
void subRegionMoveTest() {
  {
    A a;
    B b = std::move(a.b); // expected-note {{'b' became 'moved-from' here}}
    a.b.foo();            // expected-warning {{Method call on a 'moved-from' object 'b'}} expected-note {{Method call on a 'moved-from' object 'b'}}
  }
  {
    A a;
    A a1 = std::move(a); // expected-note {{Calling move constructor for 'A'}} expected-note {{Returning from move constructor for 'A'}}
    a.b.foo();           // expected-warning {{Method call on a 'moved-from' object 'b'}} expected-note {{Method call on a 'moved-from' object 'b'}}
  }
  // Don't report a misuse if any SuperRegion is already reported.
  {
    A a;
    A a1 = std::move(a); // expected-note {{'a' became 'moved-from' here}}
    a.foo();             // expected-warning {{Method call on a 'moved-from' object 'a'}} expected-note {{Method call on a 'moved-from' object 'a'}}
    a.b.foo();           // no-warning
  }
  {
    C c;
    C c1 = std::move(c); // expected-note {{'c' became 'moved-from' here}}
    c.foo();             // expected-warning {{Method call on a 'moved-from' object 'c'}} expected-note {{Method call on a 'moved-from' object 'c'}}
    c.b.foo();           // no-warning
  }
}

void resetSuperClass() {
  C c;
  C c1 = std::move(c);
  c.clear();
  C c2 = c; // no-warning
}

void reportSuperClass() {
  C c;
  C c1 = std::move(c); // expected-note {{'c' became 'moved-from' here}}
  c.foo();             // expected-warning {{Method call on a 'moved-from' object 'c'}} expected-note {{Method call on a 'moved-from' object 'c'}}
  C c2 = c;            // no-warning
}
