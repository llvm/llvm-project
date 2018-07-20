// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -DI386 -analyzer-checker=core,debug.ExprInspection -fobjc-arc -analyzer-config c++-inlining=constructors -Wno-null-dereference -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -DI386 -analyzer-checker=core,debug.ExprInspection -fobjc-arc -analyzer-config c++-inlining=constructors -Wno-null-dereference -std=c++11 -verify -DTEST_INLINABLE_ALLOCATORS %s
// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin12 -analyzer-checker=core,debug.ExprInspection -fobjc-arc -analyzer-config c++-inlining=constructors -Wno-null-dereference -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin12 -analyzer-checker=core,debug.ExprInspection -fobjc-arc -analyzer-config c++-inlining=constructors -Wno-null-dereference -std=c++11 -verify -DTEST_INLINABLE_ALLOCATORS %s

#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_eval(bool);
void clang_analyzer_checkInlined(bool);

// A simplified version of std::move.
template <typename T>
T &&move(T &obj) {
  return static_cast<T &&>(obj);
}


struct Wrapper {
  __strong id obj;
};

void test() {
  Wrapper w;
  // force a diagnostic
  *(char *)0 = 1; // expected-warning{{Dereference of null pointer}}
}


struct IntWrapper {
  int x;
};

void testCopyConstructor() {
  IntWrapper a;
  a.x = 42;

  IntWrapper b(a);
  clang_analyzer_eval(b.x == 42); // expected-warning{{TRUE}}
}

struct NonPODIntWrapper {
  int x;

  virtual int get();
};

void testNonPODCopyConstructor() {
  NonPODIntWrapper a;
  a.x = 42;

  NonPODIntWrapper b(a);
  clang_analyzer_eval(b.x == 42); // expected-warning{{TRUE}}
}


namespace ConstructorVirtualCalls {
  class A {
  public:
    int *out1, *out2, *out3;

    virtual int get() { return 1; }

    A(int *out1) {
      *out1 = get();
    }
  };

  class B : public A {
  public:
    virtual int get() { return 2; }

    B(int *out1, int *out2) : A(out1) {
      *out2 = get();
    }
  };

  class C : public B {
  public:
    virtual int get() { return 3; }

    C(int *out1, int *out2, int *out3) : B(out1, out2) {
      *out3 = get();
    }
  };

  void test() {
    int a, b, c;

    C obj(&a, &b, &c);
    clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(b == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(c == 3); // expected-warning{{TRUE}}

    clang_analyzer_eval(obj.get() == 3); // expected-warning{{TRUE}}

    // Sanity check for devirtualization.
    A *base = &obj;
    clang_analyzer_eval(base->get() == 3); // expected-warning{{TRUE}}
  }
}

namespace TemporaryConstructor {
  class BoolWrapper {
  public:
    BoolWrapper() {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
      value = true;
    }
    bool value;
  };

  void test() {
    // PR13717 - Don't crash when a CXXTemporaryObjectExpr is inlined.
    if (BoolWrapper().value)
      return;
  }
}


namespace ConstructorUsedAsRValue {
  using TemporaryConstructor::BoolWrapper;

  bool extractValue(BoolWrapper b) {
    return b.value;
  }

  void test() {
    bool result = extractValue(BoolWrapper());
    clang_analyzer_eval(result); // expected-warning{{TRUE}}
  }
}

namespace PODUninitialized {
  class POD {
  public:
    int x, y;
  };

  class PODWrapper {
  public:
    POD p;
  };

  class NonPOD {
  public:
    int x, y;

    NonPOD() {}
    NonPOD(const NonPOD &Other)
      : x(Other.x), y(Other.y) // expected-warning {{undefined}}
    {
    }
    NonPOD(NonPOD &&Other)
    : x(Other.x), y(Other.y) // expected-warning {{undefined}}
    {
    }

    NonPOD &operator=(const NonPOD &Other)
    {
      x = Other.x;
      y = Other.y; // expected-warning {{undefined}}
      return *this;
    }
    NonPOD &operator=(NonPOD &&Other)
    {
      x = Other.x;
      y = Other.y; // expected-warning {{undefined}}
      return *this;
    }
  };

  class NonPODWrapper {
  public:
    class Inner {
    public:
      int x, y;

      Inner() {}
      Inner(const Inner &Other)
        : x(Other.x), y(Other.y) // expected-warning {{undefined}}
      {
      }
      Inner(Inner &&Other)
      : x(Other.x), y(Other.y) // expected-warning {{undefined}}
      {
      }

      Inner &operator=(const Inner &Other)
      {
        x = Other.x; // expected-warning {{undefined}}
        y = Other.y;
        return *this;
      }
      Inner &operator=(Inner &&Other)
      {
        x = Other.x; // expected-warning {{undefined}}
        y = Other.y;
        return *this;
      }
    };

    Inner p;
  };

  void testPOD(const POD &pp) {
    POD p;
    p.x = 1;
    POD p2 = p; // no-warning
    clang_analyzer_eval(p2.x == 1); // expected-warning{{TRUE}}
    POD p3 = move(p); // no-warning
    clang_analyzer_eval(p3.x == 1); // expected-warning{{TRUE}}

    // Use rvalues as well.
    clang_analyzer_eval(POD(p3).x == 1); // expected-warning{{TRUE}}

    // Copy from symbolic references correctly.
    POD p4 = pp;
    // Make sure that p4.x contains a symbol after copy.
    if (p4.x > 0)
      clang_analyzer_eval(p4.x > 0); // expected-warning{{TRUE}}
    // FIXME: Element region gets in the way, so these aren't the same symbols
    // as they should be.
    clang_analyzer_eval(pp.x == p4.x); // expected-warning{{UNKNOWN}}

    PODWrapper w;
    w.p.y = 1;
    PODWrapper w2 = w; // no-warning
    clang_analyzer_eval(w2.p.y == 1); // expected-warning{{TRUE}}
    PODWrapper w3 = move(w); // no-warning
    clang_analyzer_eval(w3.p.y == 1); // expected-warning{{TRUE}}

    // Use rvalues as well.
    clang_analyzer_eval(PODWrapper(w3).p.y == 1); // expected-warning{{TRUE}}
  }

  void testNonPOD() {
    NonPOD p;
    p.x = 1;
    NonPOD p2 = p;
  }

  void testNonPODMove() {
    NonPOD p;
    p.x = 1;
    NonPOD p2 = move(p);
  }

  void testNonPODWrapper() {
    NonPODWrapper w;
    w.p.y = 1;
    NonPODWrapper w2 = w;
  }

  void testNonPODWrapperMove() {
    NonPODWrapper w;
    w.p.y = 1;
    NonPODWrapper w2 = move(w);
  }

  // Not strictly about constructors, but trivial assignment operators should
  // essentially work the same way.
  namespace AssignmentOperator {
    void testPOD() {
      POD p;
      p.x = 1;
      POD p2;
      p2 = p; // no-warning
      clang_analyzer_eval(p2.x == 1); // expected-warning{{TRUE}}
      POD p3;
      p3 = move(p); // no-warning
      clang_analyzer_eval(p3.x == 1); // expected-warning{{TRUE}}

      PODWrapper w;
      w.p.y = 1;
      PODWrapper w2;
      w2 = w; // no-warning
      clang_analyzer_eval(w2.p.y == 1); // expected-warning{{TRUE}}
      PODWrapper w3;
      w3 = move(w); // no-warning
      clang_analyzer_eval(w3.p.y == 1); // expected-warning{{TRUE}}
    }

    void testReturnValue() {
      POD p;
      p.x = 1;
      POD p2;
      clang_analyzer_eval(&(p2 = p) == &p2); // expected-warning{{TRUE}}

      PODWrapper w;
      w.p.y = 1;
      PODWrapper w2;
      clang_analyzer_eval(&(w2 = w) == &w2); // expected-warning{{TRUE}}
    }

    void testNonPOD() {
      NonPOD p;
      p.x = 1;
      NonPOD p2;
      p2 = p;
    }

    void testNonPODMove() {
      NonPOD p;
      p.x = 1;
      NonPOD p2;
      p2 = move(p);
    }

    void testNonPODWrapper() {
      NonPODWrapper w;
      w.p.y = 1;
      NonPODWrapper w2;
      w2 = w;
    }

    void testNonPODWrapperMove() {
      NonPODWrapper w;
      w.p.y = 1;
      NonPODWrapper w2;
      w2 = move(w);
    }
  }
}

namespace ArrayMembers {
  struct Primitive {
    int values[3];
  };

  void testPrimitive() {
    Primitive a = { { 1, 2, 3 } };

    clang_analyzer_eval(a.values[0] == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1] == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[2] == 3); // expected-warning{{TRUE}}

    Primitive b = a;

    clang_analyzer_eval(b.values[0] == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(b.values[1] == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(b.values[2] == 3); // expected-warning{{TRUE}}

    Primitive c;
    c = b;

    clang_analyzer_eval(c.values[0] == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(c.values[1] == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(c.values[2] == 3); // expected-warning{{TRUE}}
  }

  struct NestedPrimitive {
    int values[2][3];
  };

  void testNestedPrimitive() {
    NestedPrimitive a = { { { 0, 0, 0 }, { 1, 2, 3 } } };

    clang_analyzer_eval(a.values[1][0] == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1][1] == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1][2] == 3); // expected-warning{{TRUE}}

    NestedPrimitive b = a;

    clang_analyzer_eval(b.values[1][0] == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(b.values[1][1] == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(b.values[1][2] == 3); // expected-warning{{TRUE}}

    NestedPrimitive c;
    c = b;

    clang_analyzer_eval(c.values[1][0] == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(c.values[1][1] == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(c.values[1][2] == 3); // expected-warning{{TRUE}}
  }

  struct POD {
    IntWrapper values[3];
  };

  void testPOD() {
    POD a = { { { 1 }, { 2 }, { 3 } } };

    clang_analyzer_eval(a.values[0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[2].x == 3); // expected-warning{{TRUE}}

    POD b = a;

    clang_analyzer_eval(b.values[0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(b.values[1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(b.values[2].x == 3); // expected-warning{{TRUE}}

    POD c;
    c = b;

    clang_analyzer_eval(c.values[0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(c.values[1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(c.values[2].x == 3); // expected-warning{{TRUE}}
  }

  struct NestedPOD {
    IntWrapper values[2][3];
  };

  void testNestedPOD() {
    NestedPOD a = { { { { 0 }, { 0 }, { 0 } }, { { 1 }, { 2 }, { 3 } } } };

    clang_analyzer_eval(a.values[1][0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1][1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1][2].x == 3); // expected-warning{{TRUE}}

    NestedPOD b = a;

    clang_analyzer_eval(b.values[1][0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(b.values[1][1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(b.values[1][2].x == 3); // expected-warning{{TRUE}}

    NestedPOD c;
    c = b;

    clang_analyzer_eval(c.values[1][0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(c.values[1][1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(c.values[1][2].x == 3); // expected-warning{{TRUE}}
  }

  struct NonPOD {
    NonPODIntWrapper values[3];
  };

  void testNonPOD() {
    NonPOD a;
    a.values[0].x = 1;
    a.values[1].x = 2;
    a.values[2].x = 3;

    clang_analyzer_eval(a.values[0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[2].x == 3); // expected-warning{{TRUE}}

    NonPOD b = a;

    clang_analyzer_eval(b.values[0].x == 1); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(b.values[1].x == 2); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(b.values[2].x == 3); // expected-warning{{UNKNOWN}}

    NonPOD c;
    c = b;

    clang_analyzer_eval(c.values[0].x == 1); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(c.values[1].x == 2); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(c.values[2].x == 3); // expected-warning{{UNKNOWN}}
  }

  struct NestedNonPOD {
    NonPODIntWrapper values[2][3];
  };

  void testNestedNonPOD() {
    NestedNonPOD a;
    a.values[0][0].x = 0;
    a.values[0][1].x = 0;
    a.values[0][2].x = 0;
    a.values[1][0].x = 1;
    a.values[1][1].x = 2;
    a.values[1][2].x = 3;

    clang_analyzer_eval(a.values[1][0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1][1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1][2].x == 3); // expected-warning{{TRUE}}

    NestedNonPOD b = a;

    clang_analyzer_eval(b.values[1][0].x == 1); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(b.values[1][1].x == 2); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(b.values[1][2].x == 3); // expected-warning{{UNKNOWN}}

    NestedNonPOD c;
    c = b;

    clang_analyzer_eval(c.values[1][0].x == 1); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(c.values[1][1].x == 2); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(c.values[1][2].x == 3); // expected-warning{{UNKNOWN}}
  }
  
  struct NonPODDefaulted {
    NonPODIntWrapper values[3];

    NonPODDefaulted() = default;
    NonPODDefaulted(const NonPODDefaulted &) = default;
    NonPODDefaulted &operator=(const NonPODDefaulted &) = default;
  };

  void testNonPODDefaulted() {
    NonPODDefaulted a;
    a.values[0].x = 1;
    a.values[1].x = 2;
    a.values[2].x = 3;

    clang_analyzer_eval(a.values[0].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[1].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(a.values[2].x == 3); // expected-warning{{TRUE}}

    NonPODDefaulted b = a;

    clang_analyzer_eval(b.values[0].x == 1); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(b.values[1].x == 2); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(b.values[2].x == 3); // expected-warning{{UNKNOWN}}

    NonPODDefaulted c;
    c = b;

    clang_analyzer_eval(c.values[0].x == 1); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(c.values[1].x == 2); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(c.values[2].x == 3); // expected-warning{{UNKNOWN}}
  }
};

namespace VirtualInheritance {
  int counter;

  struct base {
    base() {
      ++counter;
    }
  };

  struct virtual_subclass : public virtual base {
    virtual_subclass() {}
  };

  struct double_subclass : public virtual_subclass {
    double_subclass() {}
  };

  void test() {
    counter = 0;
    double_subclass obj;
    clang_analyzer_eval(counter == 1); // expected-warning{{TRUE}}
  }

  struct double_virtual_subclass : public virtual virtual_subclass {
    double_virtual_subclass() {}
  };

  void testVirtual() {
    counter = 0;
    double_virtual_subclass obj;
    clang_analyzer_eval(counter == 1); // expected-warning{{TRUE}}
  }
}

namespace ZeroInitialization {
  struct raw_pair {
    int p1;
    int p2;
  };

  void testVarDecl() {
    raw_pair p{};
    clang_analyzer_eval(p.p1 == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(p.p2 == 0); // expected-warning{{TRUE}}
  }

  void testTemporary() {
    clang_analyzer_eval(raw_pair().p1 == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(raw_pair().p2 == 0); // expected-warning{{TRUE}}
  }

  void testArray() {
    raw_pair p[2] = {};
    clang_analyzer_eval(p[0].p1 == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(p[0].p2 == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(p[1].p1 == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(p[1].p2 == 0); // expected-warning{{TRUE}}
  }

  void testNew() {
    raw_pair *pp = new raw_pair();
    clang_analyzer_eval(pp->p1 == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(pp->p2 == 0); // expected-warning{{TRUE}}
  }

  void testArrayNew() {
    // FIXME: Pending proper implementation of constructors for 'new[]'.
    raw_pair *p = new raw_pair[2]();
    clang_analyzer_eval(p[0].p1 == 0); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(p[0].p2 == 0); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(p[1].p1 == 0); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(p[1].p2 == 0); // expected-warning{{UNKNOWN}}
  }

  struct initializing_pair {
  public:
    int x;
    raw_pair y;
    initializing_pair() : x(), y() {}
  };
  
  void testFieldInitializers() {
    initializing_pair p;
    clang_analyzer_eval(p.x == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(p.y.p1 == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(p.y.p2 == 0); // expected-warning{{TRUE}}
  }

  struct subclass : public raw_pair {
    subclass() = default;
  };

  void testSubclass() {
    subclass p;
    clang_analyzer_eval(p.p1 == 0); // expected-warning{{garbage}}
  }

  struct initializing_subclass : public raw_pair {
    initializing_subclass() : raw_pair() {}
  };

  void testInitializingSubclass() {
    initializing_subclass p;
    clang_analyzer_eval(p.p1 == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(p.p2 == 0); // expected-warning{{TRUE}}
  }

  struct pair_wrapper {
    pair_wrapper() : p() {}
    raw_pair p;
  };

  struct virtual_subclass : public virtual pair_wrapper {
    virtual_subclass() {}
  };

  struct double_virtual_subclass : public virtual_subclass {
    double_virtual_subclass() {
      // This previously caused a crash because the pair_wrapper subobject was
      // initialized twice.
    }
  };

  class Empty {
  public:
    static int glob;
    Empty(); // No body.
    Empty(int x); // Body below.
  };

  class PairContainer : public Empty {
  public:
    raw_pair p;
    int q;
    PairContainer() : Empty(), p() {
      // This previously caused a crash because the empty base class looked
      // like an initialization of 'p'.
    }
    PairContainer(int) : Empty(), p() {
      // Test inlining something else here.
    }
    PairContainer(double): Empty(1), p() {
      clang_analyzer_eval(p.p1 == 0); // expected-warning{{TRUE}}
      clang_analyzer_eval(p.p2 == 0); // expected-warning{{TRUE}}

      clang_analyzer_eval(q == 1); // expected-warning{{TRUE}}

      // This one's indeed UNKNOWN. Definitely not TRUE.
      clang_analyzer_eval(p.p2 == glob); // expected-warning{{UNKNOWN}}
    }
  };

  Empty::Empty(int x) {
    static_cast<PairContainer *>(this)->p.p1 = x;
    static_cast<PairContainer *>(this)->q = x;
    // Our static member will store the old garbage values of fields that aren't
    // yet initialized. It's not certainly garbage though (i.e. the constructor
    // could have been called on an initialized piece of memory), so no
    // uninitialized value warning here, and it should be a symbol, not
    // undefined value, for later comparison.
    glob = static_cast<PairContainer *>(this)->p.p2;
  }

	class Empty2 {
	public:
		static int glob_p1, glob_p2;
		Empty2(); // Body below.
	};

	class PairDoubleEmptyContainer: public Empty, public Empty2 {
	public:
    raw_pair p;
		PairDoubleEmptyContainer(): Empty(), Empty2(), p() {
      clang_analyzer_eval(p.p1 == 0); // expected-warning{{TRUE}}
      clang_analyzer_eval(p.p2 == 0); // expected-warning{{TRUE}}

      // This is indeed UNKNOWN.
      clang_analyzer_eval(p.p1 == glob_p1); // expected-warning{{UNKNOWN}}
      clang_analyzer_eval(p.p2 == glob_p2); // expected-warning{{UNKNOWN}}
		}
	};

	Empty2::Empty2() {
    glob_p1 = static_cast<PairDoubleEmptyContainer *>(this)->p.p1;
    glob_p2 = static_cast<PairDoubleEmptyContainer *>(this)->p.p2;
	}

  class PairContainerContainer {
    int padding;
    PairContainer pc;
  public:
    PairContainerContainer() : pc(1) {}
  };
}

namespace InitializerList {
  struct List {
    bool usedInitializerList;

    List() : usedInitializerList(false) {}
    List(std::initializer_list<int>) : usedInitializerList(true) {}
  };

  void testStatic() {
    List defaultCtor;
    clang_analyzer_eval(!defaultCtor.usedInitializerList); // expected-warning{{TRUE}}

    List list{1, 2};
    clang_analyzer_eval(list.usedInitializerList); // expected-warning{{TRUE}}
  }

  void testDynamic() {
    List *list = new List{1, 2};
    clang_analyzer_eval(list->usedInitializerList); // expected-warning{{TRUE}}
  }
}

namespace PR19579 {
  class C {};

  void f() {
    C();
    int a;

    extern void use(int);
    use(a); // expected-warning{{uninitialized}}
  }

  void g() {
    struct S {
      C c;
      int i;
    };
    
    // This order triggers the initialization of the inner "a" after the
    // constructor for "C" is run, which used to confuse the analyzer
    // (is "C()" the initialization of "a"?).
    struct S s = {
      C(),
      ({
        int a, b = 0;
        0;
      })
    };
  }
}

namespace NoCrashOnEmptyBaseOptimization {
  struct NonEmptyBase {
    int X;
    explicit NonEmptyBase(int X) : X(X) {}
  };

  struct EmptyBase {};

  struct S : NonEmptyBase, EmptyBase {
    S() : NonEmptyBase(0), EmptyBase() {}
  };

  void testSCtorNoCrash() {
    S s;
  }
}

namespace EmptyBaseAssign {
struct B1 {};
struct B2 { int x; };
struct D: public B1, public B2 {
const D &operator=(const D &d) {
  *((B2 *)this) = d;
  *((B1 *)this) = d;
  return *this;
}
};

void test() {
  D d1;
  d1.x = 1;
  D d2;
  d2 = d1;
  clang_analyzer_eval(d2.x == 1); // expected-warning{{TRUE}}
}
}

namespace vbase_zero_init {
class A {
  virtual void foo();
};

class B {
  virtual void bar();
public:
  static int glob_y, glob_z, glob_w;
  int x;
  B(); // Body below.
};

class C : virtual public A {
public:
  int y;
};

class D : public B, public C {
public:
  // 'z', unlike 'w', resides in an area that would have been within padding of
  // base class 'C' if it wasn't part of 'D', but only on 64-bit systems.
  int z, w;
  // Initialization order: A(), B(), C().
  D() : A(), C() {
    clang_analyzer_eval(x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(y == 0); // expected-warning{{TRUE}}
#ifdef I386
    clang_analyzer_eval(z == 3); // expected-warning{{TRUE}}
#else
    // FIXME: Should be TRUE. Initialized in B().
    clang_analyzer_eval(z == 3); // expected-warning{{UNKNOWN}}
#endif
    clang_analyzer_eval(w == 4); // expected-warning{{TRUE}}

    // FIXME: Should be UNKNOWN. Changed in B() since glob_y was assigned.
    clang_analyzer_eval(y == glob_y); // expected-warning{{TRUE}}

#ifdef I386
    clang_analyzer_eval(z == glob_z); // expected-warning{{UNKNOWN}}
#else
    // FIXME: Should be UNKNOWN. Changed in B() since glob_z was assigned.
    clang_analyzer_eval(z == glob_z); // expected-warning{{TRUE}}
#endif

    clang_analyzer_eval(w == glob_w); // expected-warning{{UNKNOWN}}
  } // no-crash
};

B::B() : x(1) {
  // Our static members will store the old garbage values of fields that aren't
  // yet initialized. These aren't certainly garbage though (i.e. the
  // constructor could have been called on an initialized piece of memory),
  // so no uninitialized value warning here, and these should be symbols, not
  // undefined values, for later comparison.
  glob_y = static_cast<D *>(this)->y;
  glob_z = static_cast<D *>(this)->z;
  glob_w = static_cast<D *>(this)->w;
  static_cast<D *>(this)->y = 2;
  static_cast<D *>(this)->z = 3;
  static_cast<D *>(this)->w = 4;
}
}
