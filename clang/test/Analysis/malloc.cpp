// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.deadcode.UnreachableCode,alpha.core.CastSize,unix.Malloc -analyzer-store=region -verify %s

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);
void *realloc(void *ptr, size_t size);
void *calloc(size_t nmemb, size_t size);
char *strdup(const char *s);

void checkThatMallocCheckerIsRunning() {
  malloc(4);
} // expected-warning{{leak}}

// Test for radar://11110132.
struct Foo {
    mutable void* m_data;
    Foo(void* data) : m_data(data) {}
};
Foo aFunction() {
    return malloc(10);
}

// Assume that functions which take a function pointer can free memory even if
// they are defined in system headers and take the const pointer to the
// allocated memory. (radar://11160612)
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