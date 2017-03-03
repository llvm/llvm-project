// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-config ipa=inlining -analyzer-config c++-allocator-inlining=true -verify %s

void clang_analyzer_eval(bool);
void clang_analyzer_checkInlined(bool);

typedef __typeof__(sizeof(int)) size_t;
extern "C" void *malloc(size_t);

// This is the standard placement new.
inline void* operator new(size_t, void* __p) throw()
{
  clang_analyzer_checkInlined(true);// expected-warning{{TRUE}}
  return __p;
}


class A {
public:
  int getZero() { return 0; }
  virtual int getNum() { return 0; }
};

void test(A &a) {
  clang_analyzer_eval(a.getZero() == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(a.getNum() == 0); // expected-warning{{UNKNOWN}}

  A copy(a);
  clang_analyzer_eval(copy.getZero() == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.getNum() == 0); // expected-warning{{TRUE}}
}


class One : public A {
public:
  virtual int getNum() { return 1; }
};

void testPathSensitivity(int x) {
  A a;
  One b;

  A *ptr;
  switch (x) {
  case 0:
    ptr = &a;
    break;
  case 1:
    ptr = &b;
    break;
  default:
    return;
  }

  // This should be true on both branches.
  clang_analyzer_eval(ptr->getNum() == x); // expected-warning {{TRUE}}
}


namespace PureVirtualParent {
  class Parent {
  public:
    virtual int pureVirtual() const = 0;
    int callVirtual() const {
      return pureVirtual();
    }
  };

  class Child : public Parent {
  public:
    virtual int pureVirtual() const {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
      return 42;
    }
  };

  void testVirtual() {
    Child x;

    clang_analyzer_eval(x.pureVirtual() == 42); // expected-warning{{TRUE}}
    clang_analyzer_eval(x.callVirtual() == 42); // expected-warning{{TRUE}}
  }
}


namespace PR13569 {
  class Parent {
  protected:
    int m_parent;
    virtual int impl() const = 0;

    Parent() : m_parent(0) {}

  public:
    int interface() const {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
      return impl();
    }
  };

  class Child : public Parent {
  protected:
    virtual int impl() const {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
      return m_parent + m_child;
    }

  public:
    Child() : m_child(0) {}

    int m_child;
  };

  void testVirtual() {
    Child x;
    x.m_child = 42;

    // Don't crash when inlining and devirtualizing.
    x.interface();
  }


  class Grandchild : public Child {};

  void testDevirtualizeToMiddle() {
    Grandchild x;
    x.m_child = 42;

    // Don't crash when inlining and devirtualizing.
    x.interface();
  }
}

namespace PR13569_virtual {
  class Parent {
  protected:
    int m_parent;
    virtual int impl() const = 0;

    Parent() : m_parent(0) {}

  public:
    int interface() const {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
      return impl();
    }
  };

  class Child : virtual public Parent {
  protected:
    virtual int impl() const {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
      return m_parent + m_child;
    }

  public:
    Child() : m_child(0) {}

    int m_child;
  };

  void testVirtual() {
    Child x;
    x.m_child = 42;

    // Don't crash when inlining and devirtualizing.
    x.interface();
  }


  class Grandchild : virtual public Child {};

  void testDevirtualizeToMiddle() {
    Grandchild x;
    x.m_child = 42;

    // Don't crash when inlining and devirtualizing.
    x.interface();
  }
}

namespace Invalidation {
  struct X {
    void touch(int &x) const {
      x = 0;
    }

    void touch2(int &x) const;

    virtual void touchV(int &x) const {
      x = 0;
    }

    virtual void touchV2(int &x) const;

    int test() const {
      // We were accidentally not invalidating under inlining
      // at one point for virtual methods with visible definitions.
      int a, b, c, d;
      touch(a);
      touch2(b);
      touchV(c);
      touchV2(d);
      return a + b + c + d; // no-warning
    }
  };
}

namespace DefaultArgs {
  int takesDefaultArgs(int i = 42) {
    return -i;
  }

  void testFunction() {
    clang_analyzer_eval(takesDefaultArgs(1) == -1); // expected-warning{{TRUE}}
    clang_analyzer_eval(takesDefaultArgs() == -42); // expected-warning{{TRUE}}
  }

  class Secret {
  public:
    static const int value = 40 + 2;
    int get(int i = value) {
      return i;
    }
  };

  void testMethod() {
    Secret obj;
    clang_analyzer_eval(obj.get(1) == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(obj.get() == 42); // expected-warning{{TRUE}}
    clang_analyzer_eval(Secret::value == 42); // expected-warning{{TRUE}}
  }

  enum ABC {
    A = 0,
    B = 1,
    C = 2
  };

  int enumUser(ABC input = B) {
    return static_cast<int>(input);
  }

  void testEnum() {
    clang_analyzer_eval(enumUser(C) == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(enumUser() == 1); // expected-warning{{TRUE}}
  }


  int exprUser(int input = 2 * 4) {
    return input;
  }

  int complicatedExprUser(int input = 2 * Secret::value) {
    return input;
  }

  void testExprs() {
    clang_analyzer_eval(exprUser(1) == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(exprUser() == 8); // expected-warning{{TRUE}}

    clang_analyzer_eval(complicatedExprUser(1) == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(complicatedExprUser() == 84); // expected-warning{{TRUE}}
  }

  int defaultReference(const int &input = 42) {
    return -input;
  }
  int defaultReferenceZero(const int &input = 0) {
    return -input;
  }

  void testReference() {
    clang_analyzer_eval(defaultReference(1) == -1); // expected-warning{{TRUE}}
    clang_analyzer_eval(defaultReference() == -42); // expected-warning{{TRUE}}

    clang_analyzer_eval(defaultReferenceZero(1) == -1); // expected-warning{{TRUE}}
    clang_analyzer_eval(defaultReferenceZero() == 0); // expected-warning{{TRUE}}
  }

  double defaultFloatReference(const double &i = 42) {
    return -i;
  }
  double defaultFloatReferenceZero(const double &i = 0) {
    return -i;
  }

  void testFloatReference() {
    clang_analyzer_eval(defaultFloatReference(1) == -1); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(defaultFloatReference() == -42); // expected-warning{{UNKNOWN}}

    clang_analyzer_eval(defaultFloatReferenceZero(1) == -1); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(defaultFloatReferenceZero() == 0); // expected-warning{{UNKNOWN}}
  }

  char defaultString(const char *s = "abc") {
    return s[1];
  }

  void testString() {
    clang_analyzer_eval(defaultString("xyz") == 'y'); // expected-warning{{TRUE}}
    clang_analyzer_eval(defaultString() == 'b'); // expected-warning{{TRUE}}
  }

  const void * const void_string = "abc";

  void testBitcastedString() {
    clang_analyzer_eval(0 != void_string); // expected-warning{{TRUE}}
    clang_analyzer_eval('b' == ((char *)void_string)[1]); // expected-warning{{TRUE}}
  }
}

namespace OperatorNew {
  class IntWrapper {
  public:
    int value;

    IntWrapper(int input) : value(input) {
      // We don't want this constructor to be inlined unless we can actually
      // use the proper region for operator new.
      // See PR12014 and <rdar://problem/12180598>.
      clang_analyzer_checkInlined(false); // no-warning
    }
  };

  void test() {
    IntWrapper *obj = new IntWrapper(42);
    // should be TRUE
    clang_analyzer_eval(obj->value == 42); // expected-warning{{UNKNOWN}}
    delete obj;
  }

  void testPlacement() {
    IntWrapper *obj = static_cast<IntWrapper *>(malloc(sizeof(IntWrapper)));
    IntWrapper *alias = new (obj) IntWrapper(42);

    clang_analyzer_eval(alias == obj); // expected-warning{{TRUE}}

    // should be TRUE
    clang_analyzer_eval(obj->value == 42); // expected-warning{{UNKNOWN}}
  }
}


namespace VirtualWithSisterCasts {
  // This entire set of tests exercises casts from sister classes and
  // from classes outside the hierarchy, which can very much confuse
  // code that uses DynamicTypeInfo or needs to construct CXXBaseObjectRegions.
  // These examples used to cause crashes in +Asserts builds.
  struct Parent {
    virtual int foo();
    int x;
  };

  struct A : Parent {
    virtual int foo() { return 42; }
  };

  struct B : Parent {
    virtual int foo();
  };

  struct Grandchild : public A {};

  struct Unrelated {};

  void testDowncast(Parent *b) {
    A *a = (A *)(void *)b;
    clang_analyzer_eval(a->foo() == 42); // expected-warning{{UNKNOWN}}

    a->x = 42;
    clang_analyzer_eval(a->x == 42); // expected-warning{{TRUE}}
  }

  void testRelated(B *b) {
    A *a = (A *)(void *)b;
    clang_analyzer_eval(a->foo() == 42); // expected-warning{{UNKNOWN}}

    a->x = 42;
    clang_analyzer_eval(a->x == 42); // expected-warning{{TRUE}}
  }

  void testUnrelated(Unrelated *b) {
    A *a = (A *)(void *)b;
    clang_analyzer_eval(a->foo() == 42); // expected-warning{{UNKNOWN}}

    a->x = 42;
    clang_analyzer_eval(a->x == 42); // expected-warning{{TRUE}}
  }

  void testCastViaNew(B *b) {
    Grandchild *g = new (b) Grandchild();
    clang_analyzer_eval(g->foo() == 42); // expected-warning{{TRUE}}

    g->x = 42;
    clang_analyzer_eval(g->x == 42); // expected-warning{{TRUE}}
  }
}


namespace QualifiedCalls {
  void test(One *object) {
    // This uses the One class from the top of the file.
    clang_analyzer_eval(object->getNum() == 1); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(object->One::getNum() == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(object->A::getNum() == 0); // expected-warning{{TRUE}}

    // getZero is non-virtual.
    clang_analyzer_eval(object->getZero() == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(object->One::getZero() == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(object->A::getZero() == 0); // expected-warning{{TRUE}}
}
}


namespace rdar12409977  {
  struct Base {
    int x;
  };

  struct Parent : public Base {
    virtual Parent *vGetThis();
    Parent *getThis() { return vGetThis(); }
  };

  struct Child : public Parent {
    virtual Child *vGetThis() { return this; }
  };

  void test() {
    Child obj;
    obj.x = 42;

    // Originally, calling a devirtualized method with a covariant return type
    // caused a crash because the return value had the wrong type. When we then
    // go to layer a CXXBaseObjectRegion on it, the base isn't a direct base of
    // the object region and we get an assertion failure.
    clang_analyzer_eval(obj.getThis()->x == 42); // expected-warning{{TRUE}}
  }
}

namespace bug16307 {
  void one_argument(int a) { }
  void call_with_less() {
    reinterpret_cast<void (*)()>(one_argument)(); // expected-warning{{Function taking 1 argument is called with fewer (0)}}
  }
}
