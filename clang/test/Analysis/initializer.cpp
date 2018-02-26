// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,cplusplus.NewDeleteLeaks,debug.ExprInspection -analyzer-config c++-inlining=constructors -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,cplusplus.NewDeleteLeaks,debug.ExprInspection -analyzer-config c++-inlining=constructors -std=c++17 -DCPLUSPLUS17 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,cplusplus.NewDeleteLeaks,debug.ExprInspection -analyzer-config c++-inlining=constructors -std=c++11 -DTEST_INLINABLE_ALLOCATORS -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,cplusplus.NewDeleteLeaks,debug.ExprInspection -analyzer-config c++-inlining=constructors -std=c++17 -DCPLUSPLUS17 -DTEST_INLINABLE_ALLOCATORS -verify %s

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

  struct HasMyStruct {
    const MyStruct &ms; // expected-note {{reference member declared here}}
    const MyStruct &msWithCleanups; // expected-note {{reference member declared here}}

    // clang's Sema issues a warning when binding a reference member to a
    // temporary value.
    HasMyStruct() : ms(5), msWithCleanups(OtherStruct(5)) {
        // expected-warning@-1 {{binding reference member 'ms' to a temporary value}}
        // expected-warning@-2 {{binding reference member 'msWithCleanups' to a temporary value}}

      // At this point the members are not garbage so we should not expect an
      // analyzer warning here even though binding a reference member
      // to a member is a terrible idea.
      ms.method(); // no-warning
      msWithCleanups.method(); // no-warning
    }
  };

  void referenceInitializeField() {
    HasMyStruct hms;
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

#ifdef CPLUSPLUS17
  C cd = {{}}; // no-crash
  const C &cdl = {{}}; // no-crash
  C &&cdr = {{}}; // no-crash

  const B &bll = {{}}; // no-crash
  const B &bcl = C({{}}); // no-crash
  B &&bcr = C({{}}); // no-crash
#endif
}
}
