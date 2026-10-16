// RUN: %clang_analyze_cc1 -w -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.deadcode.UnreachableCode \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=cplusplus.NewDelete \
// RUN:   -analyzer-checker=optin.taint.TaintPropagation \
// RUN:   -analyzer-checker=optin.taint.TaintedAlloc

// RUN: %clang_analyze_cc1 -w -verify %s \
// RUN:   -triple i386-unknown-linux-gnu \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.deadcode.UnreachableCode \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=cplusplus.NewDelete \
// RUN:   -analyzer-checker=optin.taint.TaintPropagation \
// RUN:   -analyzer-checker=optin.taint.TaintedAlloc

// RUN: %clang_analyze_cc1 -w -verify %s -DTEST_INLINABLE_ALLOCATORS \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.deadcode.UnreachableCode \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=cplusplus.NewDelete \
// RUN:   -analyzer-checker=optin.taint.TaintPropagation \
// RUN:   -analyzer-checker=optin.taint.TaintedAlloc

// RUN: %clang_analyze_cc1 -w -verify %s -DTEST_INLINABLE_ALLOCATORS \
// RUN:   -triple i386-unknown-linux-gnu \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.deadcode.UnreachableCode \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=cplusplus.NewDelete \
// RUN:   -analyzer-checker=optin.taint.TaintPropagation \
// RUN:   -analyzer-checker=optin.taint.TaintedAlloc

#include "Inputs/system-header-simulator-cxx.h"

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);
void *realloc(void *ptr, size_t size);
void *calloc(size_t nmemb, size_t size);
char *strdup(const char *s);
int scanf( const char* format, ... );

void taintAlloc() {
  size_t size = 0;
  scanf("%zu", &size);
  int *ptr = new int[size];// expected-warning{{Memory allocation function is called with a tainted (potentially attacker controlled) value}}
  delete[] ptr;
}

void checkThatMallocCheckerIsRunning() {
  malloc(4);
} // expected-warning{{leak}}

struct Foo {
    mutable void* m_data;
    Foo(void* data) : m_data(data) {}
};
Foo aFunction() {
    return malloc(10);
}

// Assume that functions which take a function pointer can free memory even if
// they are defined in system headers and take the const pointer to the
// allocated memory.
// Test default parameter.
int const_ptr_and_callback_def_param(int, const char*, int n, void(*)(void*) = free);
void r11160612_3() {
  char *x = (char*)malloc(12);
  const_ptr_and_callback_def_param(0, x, 12);
}

int const_ptr_and_callback_def_param_null(int, const char*, int n, void(*)(void*) = 0);
void r11160612_no_callback() {
  char *x = (char*)malloc(12);
  const_ptr_and_callback_def_param_null(0, x, 12);
} // expected-warning{{leak}}

// Test member function pointer.
struct CanFreeMemory {
  static void myFree(void*);
};
//This is handled because we look at the type of the parameter(not argument).
void r11160612_3(CanFreeMemory* p) {
  char *x = (char*)malloc(12);
  const_ptr_and_callback_def_param(0, x, 12, p->myFree);
}


namespace PR13751 {
  class OwningVector {
    void **storage;
    size_t length;
  public:
    OwningVector();
    ~OwningVector();
    void push_back(void *Item) {
      storage[length++] = Item;
    }
  };

  void testDestructors() {
    OwningVector v;
    v.push_back(malloc(4));
    // no leak warning; freed in destructor
  }
}

struct X { void *a; };

struct X get() {
  struct X result;
  result.a = malloc(4);
  return result; // no-warning
}

// Ensure that regions accessible through a LazyCompoundVal trigger region escape.
// Malloc checker used to report leaks for the following two test cases.
struct Property {
  char* getterName;
  Property(char* n)
  : getterName(n) {}

};
void append(Property x);

void appendWrapper(char *getterName) {
  append(Property(getterName));
}
void foo(const char* name) {
  char* getterName = strdup(name);
  appendWrapper(getterName); // no-warning
}

struct NestedProperty {
  Property prop;
  NestedProperty(Property p)
  : prop(p) {}
};
void appendNested(NestedProperty x);

void appendWrapperNested(char *getterName) {
  appendNested(NestedProperty(Property(getterName)));
}
void fooNested(const char* name) {
  char* getterName = strdup(name);
  appendWrapperNested(getterName); // no-warning
}

namespace PR31226 {
  struct b2 {
    int f;
  };

  struct b1 : virtual b2 {
    void m();
  };

  struct d : b1, b2 {
  };

  void f() {
    d *p = new d();
    p->m(); // no-crash // no-warning
  }
}

// Allow __cxa_demangle to escape.
char* test_cxa_demangle(const char* sym) {
  size_t funcnamesize = 256;
  char* funcname = (char*)malloc(funcnamesize);
  int status;
  char* ret = abi::__cxa_demangle(sym, funcname, &funcnamesize, &status);
  if (status == 0) {
    funcname = ret;
  }
  return funcname; // no-warning
}

namespace argument_leak {
class A {
  char *name;

public:
  char *getName() {
    if (!name) {
      name = static_cast<char *>(malloc(10));
    }
    return name;
  }
  ~A() {
    if (name) {
      delete[] name;
    }
  }
};

void test(A a) {
  (void)a.getName();
}
} // namespace argument_leak

#define ZERO_SIZE_PTR ((void *)16)

void test_delete_ZERO_SIZE_PTR() {
  int *Ptr = (int *)ZERO_SIZE_PTR;
  // ZERO_SIZE_PTR is specially handled but only for malloc family
  delete Ptr; // expected-warning{{Argument to 'delete' is a constant address (16)}}
}

namespace pr46253_class {
class a {
  void *realloc(int, bool = false) { realloc(1); } // no-crash
};
} // namespace pr46253_class

namespace pr46253_retty{
void realloc(void *ptr, size_t size) { realloc(ptr, size); } // no-crash
} // namespace pr46253_retty

namespace pr46253_paramty{
void *realloc(void **ptr, size_t size) { realloc(ptr, size); } // no-crash
} // namespace pr46253_paramty

namespace pr46253_paramty2{
void *realloc(void *ptr, int size) { realloc(ptr, size); } // no-crash
} // namespace pr46253_paramty2

namespace pr81597 {
struct S {};
struct T {
  void free(const S& s);
};
void f(T& t) {
  S s;
  t.free(s); // no-warning: This is not the free you are looking for...
}
} // namespace pr81597
