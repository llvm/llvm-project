// RUN: %clang_analyze_cc1 -w -verify %s\
// RUN:   -analyzer-checker=core,unix.Malloc,cplusplus.NewDeleteLeaks\
// RUN:   -analyzer-checker=debug.ExprInspection -std=c++11
// RUN: %clang_analyze_cc1 -w -verify %s\
// RUN:   -analyzer-checker=core,unix.Malloc,cplusplus.NewDeleteLeaks\
// RUN:   -analyzer-checker=debug.ExprInspection -std=c++17
// RUN: %clang_analyze_cc1 -w -verify %s\
// RUN:   -analyzer-checker=core,unix.Malloc,cplusplus.NewDeleteLeaks\
// RUN:   -analyzer-checker=debug.ExprInspection -std=c++11\
// RUN:   -DTEST_INLINABLE_ALLOCATORS
// RUN: %clang_analyze_cc1 -w -verify %s\
// RUN:   -analyzer-checker=core,unix.Malloc,cplusplus.NewDeleteLeaks\
// RUN:   -analyzer-checker=debug.ExprInspection -std=c++17\
// RUN:   -DTEST_INLINABLE_ALLOCATORS

void clang_analyzer_eval(bool);

#include "Inputs/system-header-simulator-cxx.h"

class A {
  int x;
public:
  A();
};

A::A() : x(0) {
  clang_analyzer_eval(x == 0); // expected-warning{{TRUE}}
}


class DirectMember {
  int x;
public:
  DirectMember(int value) : x(value) {}

  int getX() { return x; }
};

void testDirectMember() {
  DirectMember obj(3);
  clang_analyzer_eval(obj.getX() == 3); // expected-warning{{TRUE}}
}


class IndirectMember {
  struct {
    int x;
  };
public:
  IndirectMember(int value) : x(value) {}

  int getX() { return x; }
};

void testIndirectMember() {
  IndirectMember obj(3);
  clang_analyzer_eval(obj.getX() == 3); // expected-warning{{TRUE}}
}


struct DelegatingConstructor {
  int x;
  DelegatingConstructor(int y) { x = y; }
  DelegatingConstructor() : DelegatingConstructor(42) {}
};

void testDelegatingConstructor() {
  DelegatingConstructor obj;
  clang_analyzer_eval(obj.x == 42); // expected-warning{{TRUE}}
}


struct RefWrapper {
  RefWrapper(int *p) : x(*p) {}
  RefWrapper(int &r) : x(r) {}
  int &x;
};

void testReferenceMember() {
  int *p = 0;
  RefWrapper X(p); // expected-warning@-7 {{Dereference of null pointer}}
}

void testReferenceMember2() {
  int *p = 0;
  RefWrapper X(*p); // expected-warning {{Forming reference to null pointer}}
}


extern "C" char *strdup(const char *);

class StringWrapper {
  char *str;
public:
  StringWrapper(const char *input) : str(strdup(input)) {} // no-warning
};


// PR15070 - Constructing a type containing a non-POD array mistakenly
// tried to perform a bind instead of relying on the CXXConstructExpr,
// which caused a cast<> failure in RegionStore.
namespace DefaultConstructorWithCleanups {
  class Element {
  public:
    int value;

    class Helper {
    public:
      ~Helper();
    };
    Element(Helper h = Helper());
  };
  class Wrapper {
  public:
    Element arr[2];

    Wrapper();
  };

  Wrapper::Wrapper() /* initializers synthesized */ {}

  int test() {
    Wrapper w;
    return w.arr[0].value; // no-warning
  }
}

namespace DefaultMemberInitializers {
  struct Wrapper {
    int value = 42;

    Wrapper() {}
    Wrapper(int x) : value(x) {}
    Wrapper(bool) {}
  };

  void test() {
    Wrapper w1;
    clang_analyzer_eval(w1.value == 42); // expected-warning{{TRUE}}

    Wrapper w2(50);
    clang_analyzer_eval(w2.value == 50); // expected-warning{{TRUE}}

    Wrapper w3(false);
    clang_analyzer_eval(w3.value == 42); // expected-warning{{TRUE}}
  }

  struct StringWrapper {
    const char s[4] = "abc";
    const char *p = "xyz";

    StringWrapper(bool) {}
  };

  void testString() {
    StringWrapper w(true);
    clang_analyzer_eval(w.s[1] == 'b'); // expected-warning{{TRUE}}
    clang_analyzer_eval(w.p[1] == 'y'); // expected-warning{{TRUE}}
  }
}

namespace ReferenceInitialization {
  struct OtherStruct {
    OtherStruct(int i);
    ~OtherStruct();
  };

  struct MyStruct {
    MyStruct(int i);
    MyStruct(OtherStruct os);

    void method() const;
  };

  void referenceInitializeLocal() {
    const MyStruct &myStruct(5);
    myStruct.method(); // no-warning
  }

  void referenceInitializeMultipleLocals() {
    const MyStruct &myStruct1(5), myStruct2(5), &myStruct3(5);
    myStruct1.method(); // no-warning
    myStruct2.method(); // no-warning
    myStruct3.method(); // no-warning
  }

  void referenceInitializeLocalWithCleanup() {
    const MyStruct &myStruct(OtherStruct(5));
    myStruct.method(); // no-warning
  }
};

namespace PR31592 {
struct C {
   C() : f("}") { } // no-crash
   const char(&f)[2];
};
}

namespace CXX_initializer_lists {
struct C {
  C(std::initializer_list<int *> list);
};
void testPointerEscapeIntoLists() {
  C empty{}; // no-crash

  // Do not warn that 'x' leaks. It might have been deleted by
  // the destructor of 'c'.
  int *x = new int;
  C c{x}; // no-warning
}

void testPassListsWithExplicitConstructors() {
  (void)(std::initializer_list<int>){12}; // no-crash
}
}

namespace CXX17_aggregate_construction {
struct A {
  A();
};

struct B: public A {
};

struct C: public B {
};

struct D: public virtual A {
};

// In C++17, classes B and C are aggregates, so they will be constructed
// without actually calling their trivial constructor. Used to crash.
void foo() {
  B b = {}; // no-crash
  const B &bl = {}; // no-crash
  B &&br = {}; // no-crash

  C c = {}; // no-crash
  const C &cl = {}; // no-crash
  C &&cr = {}; // no-crash

  D d = {}; // no-crash

#if __cplusplus >= 201703L
  C cd = {{}}; // no-crash
  const C &cdl = {{}}; // no-crash
  C &&cdr = {{}}; // no-crash

  const B &bll = {{}}; // no-crash
  const B &bcl = C({{}}); // no-crash
  B &&bcr = C({{}}); // no-crash
#endif
}
} // namespace CXX17_aggregate_construction

namespace CXX17_transparent_init_list_exprs {
class A {};

class B: private A {};

B boo();
void foo1() {
  B b { boo() }; // no-crash
}

class C: virtual public A {};

C coo();
void foo2() {
  C c { coo() }; // no-crash
}

B foo_recursive() {
  B b { foo_recursive() };
}
} // namespace CXX17_transparent_init_list_exprs

namespace skip_vbase_initializer_side_effects {
int glob;
struct S {
  S() { ++glob; }
};

struct A {
  A() {}
  A(S s) {}
};

struct B : virtual A {
  B() : A(S()) {}
};

struct C : B {
  C() {}
};

void foo() {
  glob = 0;
  B b;
  clang_analyzer_eval(glob == 1); // expected-warning{{TRUE}}
  C c; // no-crash
  clang_analyzer_eval(glob == 1); // expected-warning{{TRUE}}
}
} // namespace skip_vbase_initializer_side_effects

namespace dont_skip_vbase_initializers_in_most_derived_class {
struct A {
  static int a;
  A() { a = 0; }
  A(int x) { a = x; }
};

struct B {
  static int b;
  B() { b = 0; }
  B(int y) { b = y; }
};

struct C : virtual A {
  C() : A(1) {}
};
struct D : C, virtual B {
  D() : B(2) {}
};

void testD() {
  D d;
  clang_analyzer_eval(A::a == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(B::b == 2); // expected-warning{{TRUE}}
}

struct E : virtual B, C {
  E() : B(2) {}
};

void testE() {
  E e;
  clang_analyzer_eval(A::a == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(B::b == 2); // expected-warning{{TRUE}}
}

struct F : virtual A, virtual B {
  F() : A(1) {}
};
struct G : F {
  G(): B(2) {}
};

void testG() {
  G g;
  clang_analyzer_eval(A::a == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(B::b == 2); // expected-warning{{TRUE}}
}

struct H : virtual B, virtual A {
  H(): A(1) {}
};
struct I : H {
  I(): B(2) {}
};

void testI() {
  I i;
  clang_analyzer_eval(A::a == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(B::b == 2); // expected-warning{{TRUE}}
}
} // namespace dont_skip_vbase_initializers_in_most_derived_class

namespace elementwise_copy_small_array_from_post_initializer_of_cctor {
struct String {
  String(const String &) {}
};

struct MatchComponent {
  unsigned numbers[2];
  String prerelease;
  MatchComponent(MatchComponent const &) = default;
};

MatchComponent get();
void consume(MatchComponent const &);

MatchComponent parseMatchComponent() {
  MatchComponent component = get();
  component.numbers[0] = 10;
  component.numbers[1] = 20;
  return component; // We should have no stack addr escape warning here.
}

void top() {
  consume(parseMatchComponent());
}
} // namespace elementwise_copy_small_array_from_post_initializer_of_cctor
