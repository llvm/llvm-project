// RUN: %clang_analyze_cc1 -std=c++11 -fblocks %s \
// RUN:   -verify=expected,newdelete \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDelete
//
// RUN: %clang_analyze_cc1 -DLEAKS -std=c++11 -fblocks %s \
// RUN:   -verify=expected,newdelete,leak \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDelete \
// RUN:   -analyzer-checker=cplusplus.NewDeleteLeaks
//
// RUN: %clang_analyze_cc1 -std=c++11 -fblocks -verify %s \
// RUN:   -verify=expected,leak \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDeleteLeaks
//
// RUN: %clang_analyze_cc1 -std=c++17 -fblocks %s \
// RUN:   -verify=expected,newdelete \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDelete
//
// RUN: %clang_analyze_cc1 -DLEAKS -std=c++17 -fblocks %s \
// RUN:   -verify=expected,newdelete,leak \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDelete \
// RUN:   -analyzer-checker=cplusplus.NewDeleteLeaks
//
// RUN: %clang_analyze_cc1 -std=c++17 -fblocks -verify %s \
// RUN:   -verify=expected,leak \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus.NewDeleteLeaks

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
} // leak-warning{{Potential leak of memory pointed to by 'p'}}

void testGlobalOpNewArray() {
  void *p = operator new[](0);
} // leak-warning{{Potential leak of memory pointed to by 'p'}}

void testGlobalNewExpr() {
  int *p = new int;
} // leak-warning{{Potential leak of memory pointed to by 'p'}}

void testGlobalNewExprArray() {
  int *p = new int[0];
} // leak-warning{{Potential leak of memory pointed to by 'p'}}

//----- Standard nothrow placement operators
void testGlobalNoThrowPlacementOpNewBeforeOverload() {
  void *p = operator new(0, std::nothrow);
} // leak-warning{{Potential leak of memory pointed to by 'p'}}

void testGlobalNoThrowPlacementExprNewBeforeOverload() {
  int *p = new(std::nothrow) int;
} // leak-warning{{Potential leak of memory pointed to by 'p'}}

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
  *p = 1; // newdelete-warning {{Use of memory allocated with size zero}}
  delete p;
}

int testUseZeroAlloc2() {
  int *p = (int *)operator new[](0);
  return p[0]; // newdelete-warning {{Use of memory allocated with size zero}}
  delete[] p;
}

void f(int);

void testUseZeroAlloc3() {
  int *p = new int[0];
  f(*p); // newdelete-warning {{Use of memory allocated with size zero}}
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
  f(p); // newdelete-warning{{Use of memory after it is freed}}
}

void testUseMiddleArgAfterDelete(int *p) {
  delete p;
  f(0, p); // newdelete-warning{{Use of memory after it is freed}}
}

void testUseLastArgAfterDelete(int *p) {
  delete p;
  f(0, 0, p); // newdelete-warning{{Use of memory after it is freed}}
}

void testUseSeveralArgsAfterDelete(int *p) {
  delete p;
  f(p, p, p); // newdelete-warning{{Use of memory after it is freed}}
}

void testUseRefArgAfterDelete(SomeClass &c) {
  delete &c;
  g(c); // newdelete-warning{{Use of memory after it is freed}}
}

void testVariadicArgAfterDelete() {
  SomeClass c;
  int *p = new int;
  delete p;
  g(c, 0, p); // newdelete-warning{{Use of memory after it is freed}}
}

void testUseMethodArgAfterDelete(int *p) {
  SomeClass *c = new SomeClass;
  delete p;
  c->f(p); // newdelete-warning{{Use of memory after it is freed}}
}

void testUseThisAfterDelete() {
  SomeClass *c = new SomeClass;
  delete c;
  c->f(0); // newdelete-warning{{Use of memory after it is freed}}
}

void testDoubleDelete() {
  int *p = new int;
  delete p;
  delete p; // newdelete-warning{{Attempt to free released memory}}
}

void testExprDeleteArg() {
  int i;
  delete &i; // newdelete-warning{{Argument to 'delete' is the address of the local variable 'i', which is not memory allocated by 'new'}}
}

void testExprDeleteArrArg() {
  int i;
  delete[] & i; // newdelete-warning{{Argument to 'delete[]' is the address of the local variable 'i', which is not memory allocated by 'new[]'}}
}

void testAllocDeallocNames() {
  int *p = new(std::nothrow) int[1];
  delete[] (++p);
  // newdelete-warning@-1{{Argument to 'delete[]' is offset by 4 bytes from the start of memory allocated by 'new[]'}}
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
    shared_ptr(const shared_ptr &other) : p(other.p), control(other.control) {
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

  template <typename T, typename... Args>
  shared_ptr<T> make_shared(Args &&...args) {
    return shared_ptr<T>(new T(static_cast<Args &&>(args)...));
  }

  void testSingle() {
    shared_ptr<int> a(new int);
    *a = 1;
  }

  void testMake() {
    shared_ptr<int> a = make_shared<int>();
    *a = 1;
  }

  void testMakeInParens() {
    shared_ptr<int> a = (make_shared<int>()); // no warn
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
  ~DerefClass() {
    int i = 0;
    x = &i;
    *x = 1;
  }
};

void testDoubleDeleteClassInstance() {
  DerefClass *foo = new DerefClass();
  delete foo;
  delete foo; // newdelete-warning {{Attempt to delete released memory}}
}

class EmptyClass{
public:
  EmptyClass() {}
  ~EmptyClass() {}
};

void testDoubleDeleteEmptyClass() {
  EmptyClass *foo = new EmptyClass();
  delete foo;
  delete foo; // newdelete-warning {{Attempt to delete released memory}}
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

template<void *allocate_fn(size_t)>
void* allocate_via_nttp(size_t n) {
  return allocate_fn(n);
}

template<void deallocate_fn(void*)>
void deallocate_via_nttp(void* ptr) {
  deallocate_fn(ptr);
}

void testNTTPNewNTTPDelete() {
  void* p = allocate_via_nttp<::operator new>(10);
  deallocate_via_nttp<::operator delete>(p);
} // no warn

void testNTTPNewDirectDelete() {
  void* p = allocate_via_nttp<::operator new>(10);
  ::operator delete(p);
} // no warn

void testDirectNewNTTPDelete() {
  void* p = ::operator new(10);
  deallocate_via_nttp<::operator delete>(p);
}

void not_free(void*) {
}

void testLeakBecauseNTTPIsNotDeallocation() {
  void* p = ::operator new(10);
  deallocate_via_nttp<not_free>(p);
}  // leak-warning{{Potential leak of memory pointed to by 'p'}}
