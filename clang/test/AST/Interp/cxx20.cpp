// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++20 -verify=ref %s

void test_alignas_operand() {
  alignas(8) char dummy;
  static_assert(__alignof(dummy) == 8);
}

constexpr int getMinus5() {
  int a = 10;
  a = -5;
  int *p = &a;
  return *p;
}
static_assert(getMinus5() == -5, "");

constexpr int assign() {
  int m = 10;
  int k = 12;

  m = (k = 20);

  return m;
}
static_assert(assign() == 20, "");


constexpr int pointerAssign() {
  int m = 10;
  int *p = &m;

  *p = 12; // modifies m

  return m;
}
static_assert(pointerAssign() == 12, "");

constexpr int pointerDeref() {
  int m = 12;
  int *p = &m;

  return *p;
}
static_assert(pointerDeref() == 12, "");

constexpr int pointerAssign2() {
  int m = 10;
  int *p = &m;
  int **pp = &p;

  **pp = 12;

  int v = **pp;

  return v;
}
static_assert(pointerAssign2() == 12, "");

constexpr int unInitLocal() {
  int a;
  return a; // ref-note {{read of uninitialized object}} \
            // expected-note {{read of object outside its lifetime}}
            // FIXME: ^^^ Wrong diagnostic.
}
static_assert(unInitLocal() == 0, ""); // ref-error {{not an integral constant expression}} \
                                       // ref-note {{in call to 'unInitLocal()'}} \
                                       // expected-error {{not an integral constant expression}} \
                                       // expected-note {{in call to 'unInitLocal()'}} \

constexpr int initializedLocal() {
  int a;
  a = 20;
  return a;
}
static_assert(initializedLocal() == 20);

constexpr int initializedLocal2() {
  int a[2];
  return *a; // expected-note {{read of object outside its lifetime}} \
             // ref-note {{read of uninitialized object is not allowed in a constant expression}}
}
static_assert(initializedLocal2() == 20); // expected-error {{not an integral constant expression}} \
                                          // expected-note {{in call to}} \
                                          // ref-error {{not an integral constant expression}} \
                                          // ref-note {{in call to}}


struct Int { int a; }; // expected-note {{subobject declared here}}
constexpr int initializedLocal3() {
  Int i; // expected-note {{subobject of type 'int' is not initialized}}
  return i.a; // ref-note {{read of uninitialized object is not allowed in a constant expression}}
}
static_assert(initializedLocal3() == 20); // expected-error {{not an integral constant expression}} \
                                          // expected-note {{in call to}} \
                                          // ref-error {{not an integral constant expression}} \
                                          // ref-note {{in call to}}



#if 0
// FIXME: This code should be rejected because we pass an uninitialized value
//   as a function parameter.
constexpr int inc(int a) { return a + 1; }
constexpr int f() {
    int i;
    return inc(i);
}
static_assert(f());
#endif

/// Distinct literals have disctinct addresses.
/// see https://github.com/llvm/llvm-project/issues/58754
constexpr auto foo(const char *p) { return p; }
constexpr auto p1 = "test1";
constexpr auto p2 = "test2";

constexpr bool b1 = foo(p1) == foo(p1);
static_assert(b1);

constexpr bool b2 = foo(p1) == foo(p2); // ref-error {{must be initialized by a constant expression}} \
                                        // ref-note {{comparison of addresses of literals}} \
                                        // ref-note {{declared here}}
static_assert(!b2); // ref-error {{not an integral constant expression}} \
                    // ref-note {{not a constant expression}}

constexpr auto name1() { return "name1"; }
constexpr auto name2() { return "name2"; }

constexpr auto b3 = name1() == name1();
static_assert(b3);
constexpr auto b4 = name1() == name2(); // ref-error {{must be initialized by a constant expression}} \
                                        // ref-note {{has unspecified value}} \
                                        // ref-note {{declared here}}
static_assert(!b4); // ref-error {{not an integral constant expression}} \
                    // ref-note {{not a constant expression}}

namespace UninitializedFields {
  class A {
  public:
    int a; // expected-note 2{{subobject declared here}} \
           // ref-note 2{{subobject declared here}}
    constexpr A() {}
  };
  constexpr A a; // expected-error {{must be initialized by a constant expression}} \
                 // expected-note {{subobject of type 'int' is not initialized}} \
                 // ref-error {{must be initialized by a constant expression}} \
                 // ref-note {{subobject of type 'int' is not initialized}}


  class Base {
  public:
    bool b;
    int a; // expected-note {{subobject declared here}} \
           // ref-note {{subobject declared here}}
    constexpr Base() : b(true) {}
  };

  class Derived : public Base {
  public:
    constexpr Derived() : Base() {} // expected-note {{subobject of type 'int' is not initialized}}
  };

constexpr Derived D; // expected-error {{must be initialized by a constant expression}} \\
                     // expected-note {{in call to 'Derived()'}} \
                     // ref-error {{must be initialized by a constant expression}} \
                     // ref-note {{subobject of type 'int' is not initialized}}

  class C2 {
  public:
    A a;
    constexpr C2() {} // expected-note {{subobject of type 'int' is not initialized}}
  };
  constexpr C2 c2; // expected-error {{must be initialized by a constant expression}} \
                   // expected-note {{in call to 'C2()'}} \
                   // ref-error {{must be initialized by a constant expression}} \
                   // ref-note {{subobject of type 'int' is not initialized}}


  // FIXME: These two are currently disabled because the array fields
  //   cannot be initialized.
#if 0
  class C3 {
  public:
    A a[2];
    constexpr C3() {}
  };
  constexpr C3 c3; // expected-error {{must be initialized by a constant expression}} \
                   // expected-note {{subobject of type 'int' is not initialized}} \
                   // ref-error {{must be initialized by a constant expression}} \
                   // ref-note {{subobject of type 'int' is not initialized}}

  class C4 {
  public:
    bool B[2][3]; // expected-note {{subobject declared here}} \
                  // ref-note {{subobject declared here}}
    constexpr C4(){}
  };
  constexpr C4 c4; // expected-error {{must be initialized by a constant expression}} \
                   // expected-note {{subobject of type 'bool' is not initialized}} \
                   // ref-error {{must be initialized by a constant expression}} \
                   // ref-note {{subobject of type 'bool' is not initialized}}
#endif
};
