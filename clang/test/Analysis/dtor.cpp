// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,debug.ExprInspection,cplusplus -analyzer-config c++-inlining=destructors -Wno-null-dereference -Wno-inaccessible-base -verify -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(bool);
void clang_analyzer_checkInlined(bool);

class A {
public:
  ~A() { 
    int *x = 0;
    *x = 3; // expected-warning{{Dereference of null pointer}}
  }
};

int main() {
  A a;
}


typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);

class SmartPointer {
  void *X;
public:
  SmartPointer(void *x) : X(x) {}
  ~SmartPointer() {
    free(X);
  }
};

void testSmartPointer() {
  char *mem = (char*)malloc(4);
  {
    SmartPointer Deleter(mem);
    // destructor called here
  }
  *mem = 0; // expected-warning{{Use of memory after it is freed}}
}


void doSomething();
void testSmartPointer2() {
  char *mem = (char*)malloc(4);
  {
    SmartPointer Deleter(mem);
    // Remove dead bindings...
    doSomething();
    // destructor called here
  }
  *mem = 0; // expected-warning{{Use of memory after it is freed}}
}


class Subclass : public SmartPointer {
public:
  Subclass(void *x) : SmartPointer(x) {}
};

void testSubclassSmartPointer() {
  char *mem = (char*)malloc(4);
  {
    Subclass Deleter(mem);
    // Remove dead bindings...
    doSomething();
    // destructor called here
  }
  *mem = 0; // expected-warning{{Use of memory after it is freed}}
}


class MultipleInheritance : public Subclass, public SmartPointer {
public:
  MultipleInheritance(void *a, void *b) : Subclass(a), SmartPointer(b) {}
};

void testMultipleInheritance1() {
  char *mem = (char*)malloc(4);
  {
    MultipleInheritance Deleter(mem, 0);
    // Remove dead bindings...
    doSomething();
    // destructor called here
  }
  *mem = 0; // expected-warning{{Use of memory after it is freed}}
}

void testMultipleInheritance2() {
  char *mem = (char*)malloc(4);
  {
    MultipleInheritance Deleter(0, mem);
    // Remove dead bindings...
    doSomething();
    // destructor called here
  }
  *mem = 0; // expected-warning{{Use of memory after it is freed}}
}

void testMultipleInheritance3() {
  char *mem = (char*)malloc(4);
  {
    MultipleInheritance Deleter(mem, mem);
    // Remove dead bindings...
    doSomething();
    // destructor called here
    // expected-warning@28 {{Attempt to free released memory}}
  }
}


class SmartPointerMember {
  SmartPointer P;
public:
  SmartPointerMember(void *x) : P(x) {}
};

void testSmartPointerMember() {
  char *mem = (char*)malloc(4);
  {
    SmartPointerMember Deleter(mem);
    // Remove dead bindings...
    doSomething();
    // destructor called here
  }
  *mem = 0; // expected-warning{{Use of memory after it is freed}}
}


struct IntWrapper {
  IntWrapper() : x(0) {}
  ~IntWrapper();
  int *x;
};

void testArrayInvalidation() {
  int i = 42;
  int j = 42;

  {
    IntWrapper arr[2];

    // There should be no undefined value warnings here.
    // Eventually these should be TRUE as well, but right now
    // we can't handle array constructors.
    clang_analyzer_eval(arr[0].x == 0); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(arr[1].x == 0); // expected-warning{{UNKNOWN}}

    arr[0].x = &i;
    arr[1].x = &j;
    clang_analyzer_eval(*arr[0].x == 42); // expected-warning{{TRUE}}
    clang_analyzer_eval(*arr[1].x == 42); // expected-warning{{TRUE}}
  }

  // The destructors should have invalidated i and j.
  clang_analyzer_eval(i == 42); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(j == 42); // expected-warning{{UNKNOWN}}
}



// Don't crash on a default argument inside an initializer.
struct DefaultArg {
  DefaultArg(int x = 0) {}
  ~DefaultArg();
};

struct InheritsDefaultArg : DefaultArg {
  InheritsDefaultArg() {}
  virtual ~InheritsDefaultArg();
};

void testDefaultArg() {
  InheritsDefaultArg a;
  // Force a bug to be emitted.
  *(char *)0 = 1; // expected-warning{{Dereference of null pointer}}
}


namespace DestructorVirtualCalls {
  class A {
  public:
    int *out1, *out2, *out3;

    virtual int get() { return 1; }

    ~A() {
      *out1 = get();
    }
  };

  class B : public A {
  public:
    virtual int get() { return 2; }

    ~B() {
      *out2 = get();
    }
  };

  class C : public B {
  public:
    virtual int get() { return 3; }

    ~C() {
      *out3 = get();
    }
  };

  void test() {
    int a, b, c;

    // New scope for the C object.
    {
      C obj;
      clang_analyzer_eval(obj.get() == 3); // expected-warning{{TRUE}}

      // Sanity check for devirtualization.
      A *base = &obj;
      clang_analyzer_eval(base->get() == 3); // expected-warning{{TRUE}}

      obj.out1 = &a;
      obj.out2 = &b;
      obj.out3 = &c;
    }

    clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(b == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(c == 3); // expected-warning{{TRUE}}
  }
}


namespace DestructorsShouldNotAffectReturnValues {
  class Dtor {
  public:
    ~Dtor() {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
    }
  };

  void *allocate() {
    Dtor d;
    return malloc(4); // no-warning
  }

  void test() {
    // At one point we had an issue where the statements inside an
    // inlined destructor kept us from finding the return statement,
    // leading the analyzer to believe that the malloc'd memory had leaked.
    void *p = allocate();
    free(p); // no-warning
  }
}

namespace MultipleInheritanceVirtualDtors {
  class VirtualDtor {
  protected:
    virtual ~VirtualDtor() {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
    }
  };

  class NonVirtualDtor {
  protected:
    ~NonVirtualDtor() {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
    }
  };

  class SubclassA : public VirtualDtor, public NonVirtualDtor {
  public:
    virtual ~SubclassA() {}
  };
  class SubclassB : public NonVirtualDtor, public VirtualDtor {
  public:
    virtual ~SubclassB() {}
  };

  void test() {
    SubclassA a;
    SubclassB b;
  }
}

namespace ExplicitDestructorCall {
  class VirtualDtor {
  public:
    virtual ~VirtualDtor() {
      clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
    }
  };
  
  class Subclass : public VirtualDtor {
  public:
    virtual ~Subclass() {
      clang_analyzer_checkInlined(false); // no-warning
    }
  };
  
  void destroy(Subclass *obj) {
    obj->VirtualDtor::~VirtualDtor();
  }
}


namespace MultidimensionalArrays {
  void testArrayInvalidation() {
    int i = 42;
    int j = 42;

    {
      IntWrapper arr[2][2];

      // There should be no undefined value warnings here.
      // Eventually these should be TRUE as well, but right now
      // we can't handle array constructors.
      clang_analyzer_eval(arr[0][0].x == 0); // expected-warning{{UNKNOWN}}
      clang_analyzer_eval(arr[1][1].x == 0); // expected-warning{{UNKNOWN}}

      arr[0][0].x = &i;
      arr[1][1].x = &j;
      clang_analyzer_eval(*arr[0][0].x == 42); // expected-warning{{TRUE}}
      clang_analyzer_eval(*arr[1][1].x == 42); // expected-warning{{TRUE}}
    }

    // The destructors should have invalidated i and j.
    clang_analyzer_eval(i == 42); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(j == 42); // expected-warning{{UNKNOWN}}
  }
}

namespace LifetimeExtension {
  struct IntWrapper {
	int x;
	IntWrapper(int y) : x(y) {}
	IntWrapper() {
      extern void use(int);
      use(x); // no-warning
	}
  };

  struct DerivedWrapper : public IntWrapper {
	DerivedWrapper(int y) : IntWrapper(y) {}
  };

  DerivedWrapper get() {
	return DerivedWrapper(1);
  }

  void test() {
	const DerivedWrapper &d = get(); // lifetime extended here
  }


  class SaveOnDestruct {
  public:
    static int lastOutput;
    int value;

    SaveOnDestruct();
    ~SaveOnDestruct() {
      lastOutput = value;
    }
  };

  void testSimple() {
    {
      const SaveOnDestruct &obj = SaveOnDestruct();
      if (obj.value != 42)
        return;
      // destructor called here
    }

    clang_analyzer_eval(SaveOnDestruct::lastOutput == 42); // expected-warning{{TRUE}}
  }

  struct NRCheck {
    bool bool_;
    NRCheck():bool_(true) {}
    ~NRCheck() __attribute__((noreturn));
    operator bool() const { return bool_; }
  };

  struct CheckAutoDestructor {
    bool bool_;
    CheckAutoDestructor():bool_(true) {}
    operator bool() const { return bool_; }
  };

  struct CheckCustomDestructor {
    bool bool_;
    CheckCustomDestructor():bool_(true) {}
    ~CheckCustomDestructor();
    operator bool() const { return bool_; }
  };

  bool testUnnamedNR() {
    if (NRCheck())
      return true;
    return false;
  }

  bool testNamedNR() {
    if (NRCheck c = NRCheck())
      return true;
    return false;
  }

  bool testUnnamedAutoDestructor() {
    if (CheckAutoDestructor())
      return true;
    return false;
  }

  bool testNamedAutoDestructor() {
    if (CheckAutoDestructor c = CheckAutoDestructor())
      return true;
    return false;
  }

  bool testUnnamedCustomDestructor() {
    if (CheckCustomDestructor())
      return true;
    return false;
  }

  // This case used to cause an unexpected "Undefined or garbage value returned
  // to caller" warning
  bool testNamedCustomDestructor() {
    if (CheckCustomDestructor c = CheckCustomDestructor())
      return true;
    return false;
  }

  bool testMultipleTemporariesCustomDestructor() {
    if (CheckCustomDestructor c = (CheckCustomDestructor(), CheckCustomDestructor()))
      return true;
    return false;
  }

  class VirtualDtorBase {
  public:
    int value;
    virtual ~VirtualDtorBase() {}
  };

  class SaveOnVirtualDestruct : public VirtualDtorBase {
  public:
    static int lastOutput;

    SaveOnVirtualDestruct();
    virtual ~SaveOnVirtualDestruct() {
      lastOutput = value;
    }
  };

  void testVirtual() {
    {
      const VirtualDtorBase &obj = SaveOnVirtualDestruct();
      if (obj.value != 42)
        return;
      // destructor called here
    }

    clang_analyzer_eval(SaveOnVirtualDestruct::lastOutput == 42); // expected-warning{{TRUE}}
  }
}

namespace NoReturn {
  struct NR {
    ~NR() __attribute__((noreturn));
  };

  void f(int **x) {
    NR nr;
  }

  void g() {
    int *x;
    f(&x);
    *x = 47; // no warning
  }

  void g2(int *x) {
    if (! x) NR();
    *x = 47; // no warning
  }
}

namespace PseudoDtor {
  template <typename T>
  void destroy(T &obj) {
    clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
    obj.~T();
  }

  void test() {
    int i;
    destroy(i);
    clang_analyzer_eval(true); // expected-warning{{TRUE}}
  }
}

namespace Incomplete {
  class Foo; // expected-note{{forward declaration}}
  void f(Foo *foo) { delete foo; } // expected-warning{{deleting pointer to incomplete type}}
}

namespace TypeTraitExpr {
template <bool IsSimple, typename T>
struct copier {
  static void do_copy(T *dest, const T *src, unsigned count);
};
template <typename T, typename U>
void do_copy(T *dest, const U *src, unsigned count) {
  const bool IsSimple = __is_trivial(T) && __is_same(T, U);
  copier<IsSimple, T>::do_copy(dest, src, count);
}
struct NonTrivial {
  int *p;
  NonTrivial() : p(new int[1]) { p[0] = 0; }
  NonTrivial(const NonTrivial &other) {
    p = new int[1];
    do_copy(p, other.p, 1);
  }
  NonTrivial &operator=(const NonTrivial &other) {
    p = other.p;
    return *this;
  }
  ~NonTrivial() {
    delete[] p; // expected-warning {{free released memory}}
  }
};

void f() {
  NonTrivial nt1;
  NonTrivial nt2(nt1);
  nt1 = nt2;
  clang_analyzer_eval(__is_trivial(NonTrivial)); // expected-warning{{FALSE}}
  clang_analyzer_eval(__alignof(NonTrivial) > 0); // expected-warning{{TRUE}}
}
}
