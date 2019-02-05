// RUN: %clang_analyze_cc1 -std=c++11 -fblocks -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDelete
//
// RUN: %clang_analyze_cc1 -DLEAKS -std=c++11 -fblocks -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDeleteLeaks
//
// RUN: %clang_analyze_cc1 -std=c++11 -fblocks -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDelete \
// RUN:   -analyzer-config c++-allocator-inlining=true
//
// RUN: %clang_analyze_cc1 -DLEAKS -std=c++11 -fblocks -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDeleteLeaks \
// RUN:   -analyzer-config c++-allocator-inlining=true
//
// RUN: %clang_analyze_cc1 -DTEST_INLINABLE_ALLOCATORS \
// RUN:   -std=c++11 -fblocks -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDelete
//
// RUN: %clang_analyze_cc1 -DLEAKS -DTEST_INLINABLE_ALLOCATORS \
// RUN:   -std=c++11 -fblocks -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDeleteLeaks
//
// RUN: %clang_analyze_cc1 -DTEST_INLINABLE_ALLOCATORS \
// RUN:   -std=c++11 -fblocks -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDelete \
// RUN:   -analyzer-config c++-allocator-inlining=true
//
// RUN: %clang_analyze_cc1 -DLEAKS -DTEST_INLINABLE_ALLOCATORS \
// RUN:   -std=c++11 -fblocks -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDeleteLeaks \
// RUN:   -analyzer-config c++-allocator-inlining=true

#include "Inputs/system-header-simulator-cxx.h"

typedef __typeof__(sizeof(int)) size_t;
extern "C" void *malloc(size_t);
extern "C" void free (void* ptr);
int *global;

//------------------
// check for leaks
//------------------

//----- Standard non-placement operators
void testGlobalOpNew() {
  void *p = operator new(0);
}
#ifdef LEAKS
// expected-warning@-2{{Potential leak of memory pointed to by 'p'}}
#endif

void testGlobalOpNewArray() {
  void *p = operator new[](0);
}
#ifdef LEAKS
// expected-warning@-2{{Potential leak of memory pointed to by 'p'}}
#endif

void testGlobalNewExpr() {
  int *p = new int;
}
#ifdef LEAKS
// expected-warning@-2{{Potential leak of memory pointed to by 'p'}}
#endif

void testGlobalNewExprArray() {
  int *p = new int[0];
}
#ifdef LEAKS
// expected-warning@-2{{Potential leak of memory pointed to by 'p'}}
#endif

//----- Standard nothrow placement operators
void testGlobalNoThrowPlacementOpNewBeforeOverload() {
  void *p = operator new(0, std::nothrow);
}
#ifdef LEAKS
#ifndef TEST_INLINABLE_ALLOCATORS
// expected-warning@-3{{Potential leak of memory pointed to by 'p'}}
#endif
#endif

void testGlobalNoThrowPlacementExprNewBeforeOverload() {
  int *p = new(std::nothrow) int;
}
#ifdef LEAKS
#ifndef TEST_INLINABLE_ALLOCATORS
// expected-warning@-3{{Potential leak of memory pointed to by 'p'}}
#endif
#endif

//----- Standard pointer placement operators
void testGlobalPointerPlacementNew() {
  int i;

  void *p1 = operator new(0, &i); // no warn

  void *p2 = operator new[](0, &i); // no warn

  int *p3 = new(&i) int; // no warn

  int *p4 = new(&i) int[0]; // no warn
}

//----- Other cases
void testNewMemoryIsInHeap() {
  int *p = new int;
  if (global != p) // condition is always true as 'p' wraps a heap region that 
                   // is different from a region wrapped by 'global'
    global = p; // pointer escapes
}

struct PtrWrapper {
  int *x;

  PtrWrapper(int *input) : x(input) {}
};

void testNewInvalidationPlacement(PtrWrapper *w) {
  // Ensure that we don't consider this a leak.
  new (w) PtrWrapper(new int); // no warn
}

//-----------------------------------------
// check for usage of zero-allocated memory
//-----------------------------------------

void testUseZeroAlloc1() {
  int *p = (int *)operator new(0);
  *p = 1; // expected-warning {{Use of zero-allocated memory}}
  delete p;
}

int testUseZeroAlloc2() {
  int *p = (int *)operator new[](0);
  return p[0]; // expected-warning {{Use of zero-allocated memory}}
  delete[] p;
}

void f(int);

void testUseZeroAlloc3() {
  int *p = new int[0];
  f(*p); // expected-warning {{Use of zero-allocated memory}}
  delete[] p;
}

//---------------
// other checks
//---------------

class SomeClass {
public:
  void f(int *p);
};

void f(int *p1, int *p2 = 0, int *p3 = 0);
void g(SomeClass &c, ...);

void testUseFirstArgAfterDelete() {
  int *p = new int;
  delete p;
  f(p); // expected-warning{{Use of memory after it is freed}}
}

void testUseMiddleArgAfterDelete(int *p) {
  delete p;
  f(0, p); // expected-warning{{Use of memory after it is freed}}
}

void testUseLastArgAfterDelete(int *p) {
  delete p;
  f(0, 0, p); // expected-warning{{Use of memory after it is freed}}
}

void testUseSeveralArgsAfterDelete(int *p) {
  delete p;
  f(p, p, p); // expected-warning{{Use of memory after it is freed}}
}

void testUseRefArgAfterDelete(SomeClass &c) {
  delete &c;
  g(c); // expected-warning{{Use of memory after it is freed}}
}

void testVariadicArgAfterDelete() {
  SomeClass c;
  int *p = new int;
  delete p;
  g(c, 0, p); // expected-warning{{Use of memory after it is freed}}
}

void testUseMethodArgAfterDelete(int *p) {
  SomeClass *c = new SomeClass;
  delete p;
  c->f(p); // expected-warning{{Use of memory after it is freed}}
}

void testUseThisAfterDelete() {
  SomeClass *c = new SomeClass;
  delete c;
  c->f(0); // expected-warning{{Use of memory after it is freed}}
}

void testDoubleDelete() {
  int *p = new int;
  delete p;
  delete p; // expected-warning{{Attempt to free released memory}}
}

void testExprDeleteArg() {
  int i;
  delete &i; // expected-warning{{Argument to 'delete' is the address of the local variable 'i', which is not memory allocated by 'new'}}
}

void testExprDeleteArrArg() {
  int i;
  delete[] &i; // expected-warning{{Argument to 'delete[]' is the address of the local variable 'i', which is not memory allocated by 'new[]'}}
}

void testAllocDeallocNames() {
  int *p = new(std::nothrow) int[1];
  delete[] (++p);
#ifndef TEST_INLINABLE_ALLOCATORS
  // expected-warning@-2{{Argument to 'delete[]' is offset by 4 bytes from the start of memory allocated by 'new[]'}}
#endif
}

//--------------------------------
// Test escape of newed const pointer. Note, a const pointer can be deleted.
//--------------------------------
struct StWithConstPtr {
  const int *memp;
};
void escape(const int &x);
void escapeStruct(const StWithConstPtr &x);
void escapePtr(const StWithConstPtr *x);
void escapeVoidPtr(const void *x);

void testConstEscape() {
  int *p = new int(1);
  escape(*p);
} // no-warning

void testConstEscapeStruct() {
  StWithConstPtr *St = new StWithConstPtr();
  escapeStruct(*St);
} // no-warning

void testConstEscapeStructPtr() {
  StWithConstPtr *St = new StWithConstPtr();
  escapePtr(St);
} // no-warning

void testConstEscapeMember() {
  StWithConstPtr St;
  St.memp = new int(2);
  escapeVoidPtr(St.memp);
} // no-warning

void testConstEscapePlacementNew() {
  int *x = (int *)malloc(sizeof(int));
  void *y = new (x) int;
  escapeVoidPtr(y);
} // no-warning

//============== Test Uninitialized delete delete[]========================
void testUninitDelete() {
  int *x;
  int * y = new int;
  delete y;
  delete x; // expected-warning{{Argument to 'delete' is uninitialized}}
}

void testUninitDeleteArray() {
  int *x;
  int * y = new int[5];
  delete[] y;
  delete[] x; // expected-warning{{Argument to 'delete[]' is uninitialized}}
}

void testUninitFree() {
  int *x;
  free(x); // expected-warning{{1st function call argument is an uninitialized value}}
}

void testUninitDeleteSink() {
  int *x;
  delete x; // expected-warning{{Argument to 'delete' is uninitialized}}
  (*(volatile int *)0 = 1); // no warn
}

void testUninitDeleteArraySink() {
  int *x;
  delete[] x; // expected-warning{{Argument to 'delete[]' is uninitialized}}
  (*(volatile int *)0 = 1); // no warn
}

namespace reference_count {
  class control_block {
    unsigned count;
  public:
    control_block() : count(0) {}
    void retain() { ++count; }
    int release() { return --count; }
  };

  template <typename T>
  class shared_ptr {
    T *p;
    control_block *control;

  public:
    shared_ptr() : p(0), control(0) {}
    explicit shared_ptr(T *p) : p(p), control(new control_block) {
      control->retain();
    }
    shared_ptr(shared_ptr &other) : p(other.p), control(other.control) {
      if (control)
          control->retain();
    }
    ~shared_ptr() {
      if (control && control->release() == 0) {
        delete p;
        delete control;
      }
    };

    T &operator *() {
      return *p;
    };

    void swap(shared_ptr &other) {
      T *tmp = p;
      p = other.p;
      other.p = tmp;

      control_block *ctrlTmp = control;
      control = other.control;
      other.control = ctrlTmp;
    }
  };

  void testSingle() {
    shared_ptr<int> a(new int);
    *a = 1;
  }

  void testDouble() {
    shared_ptr<int> a(new int);
    shared_ptr<int> b = a;
    *a = 1;
  }

  void testInvalidated() {
    shared_ptr<int> a(new int);
    shared_ptr<int> b = a;
    *a = 1;

    extern void use(shared_ptr<int> &);
    use(b);
  }

  void testNestedScope() {
    shared_ptr<int> a(new int);
    {
      shared_ptr<int> b = a;
    }
    *a = 1;
  }

  void testSwap() {
    shared_ptr<int> a(new int);
    shared_ptr<int> b;
    shared_ptr<int> c = a;
    shared_ptr<int>(c).swap(b);
  }

  void testUseAfterFree() {
    int *p = new int;
    {
      shared_ptr<int> a(p);
      shared_ptr<int> b = a;
    }

    // FIXME: We should get a warning here, but we don't because we've
    // conservatively modeled ~shared_ptr.
    *p = 1;
  }
}

// Test double delete
class DerefClass{
public:
  int *x;
  DerefClass() {}
  ~DerefClass() {*x = 1;}
};

void testDoubleDeleteClassInstance() {
  DerefClass *foo = new DerefClass();
  delete foo;
  delete foo; // expected-warning {{Attempt to delete released memory}}
}

class EmptyClass{
public:
  EmptyClass() {}
  ~EmptyClass() {}
};

void testDoubleDeleteEmptyClass() {
  EmptyClass *foo = new EmptyClass();
  delete foo;
  delete foo;  // expected-warning {{Attempt to delete released memory}}
}

struct Base {
  virtual ~Base() {}
};

struct Derived : Base {
};

Base *allocate() {
  return new Derived;
}

void shouldNotReportLeak() {
  Derived *p = (Derived *)allocate();
  delete p;
}
