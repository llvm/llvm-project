// RUN: %check_clang_tidy %s readability-non-const-parameter %t

// Currently the checker only warns about pointer arguments.
//
// It can be defined both that the data is const and that the pointer is const,
// the checker only checks if the data can be const-specified.
//
// It does not warn about pointers to records or function pointers.

// Some external function where first argument is nonconst and second is const.
char *strcpy1(char *dest, const char *src);
unsigned my_strcpy(char *buf, const char *s);
unsigned my_strlen(const char *buf);

// CHECK-MESSAGES: :[[@LINE+1]]:29: warning: pointer parameter 'last' can be pointer to const [readability-non-const-parameter]
void warn1(int *first, int *last) {
  // CHECK-FIXES: void warn1(int *first, const int *last) {
  *first = 0;
  if (first < last) {
  } // <- last can be const
}

// TODO: warning should be written here
void warn2(char *p) {
  char buf[10];
  strcpy1(buf, p);
}

// CHECK-MESSAGES: :[[@LINE+1]]:19: warning: pointer parameter 'p' can be
void assign1(int *p) {
  // CHECK-FIXES: void assign1(const int *p) {
  const int *q;
  q = p;
}

// CHECK-MESSAGES: :[[@LINE+1]]:19: warning: pointer parameter 'p' can be
void assign2(int *p) {
  // CHECK-FIXES: void assign2(const int *p) {
  const int *q;
  q = p + 1;
}

void assign3(int *p) {
  *p = 0;
}

void assign4(int *p) {
  *p += 2;
}

void assign5(char *p) {
  p[0] = 0;
}

void assign6(int *p) {
  int *q;
  q = p++;
}

void assign7(char *p) {
  char *a, *b;
  a = b = p;
}

void assign8(char *a, char *b) {
  char *x;
  x = (a ? a : b);
}

void assign9(unsigned char *str, const unsigned int i) {
  unsigned char *p;
  for (p = str + i; *p;) {
  }
}

void assign10(int *buf) {
  int i, *p;
  for (i = 0, p = buf; i < 10; i++, p++) {
    *p = 1;
  }
}

// CHECK-MESSAGES: :[[@LINE+1]]:17: warning: pointer parameter 'p' can be
void init1(int *p) {
  // CHECK-FIXES: void init1(const int *p) {
  const int *q = p;
}

// CHECK-MESSAGES: :[[@LINE+1]]:17: warning: pointer parameter 'p' can be
void init2(int *p) {
  // CHECK-FIXES: void init2(const int *p) {
  const int *q = p + 1;
}

void init3(int *p) {
  int *q = p;
}

void init4(float *p) {
  int *q = (int *)p;
}

void init5(int *p) {
  int *i = p ? p : 0;
}

void init6(int *p) {
  int *a[] = {p, p, 0};
}

void init7(int *p, int x) {
  for (int *q = p + x - 1; 0; q++)
    ;
}

// CHECK-MESSAGES: :[[@LINE+1]]:18: warning: pointer parameter 'p' can be
int return1(int *p) {
  // CHECK-FIXES: int return1(const int *p) {
  return *p;
}

// CHECK-MESSAGES: :[[@LINE+1]]:25: warning: pointer parameter 'p' can be
const int *return2(int *p) {
  // CHECK-FIXES: const int *return2(const int *p) {
  return p;
}

// CHECK-MESSAGES: :[[@LINE+1]]:25: warning: pointer parameter 'p' can be
const int *return3(int *p) {
  // CHECK-FIXES: const int *return3(const int *p) {
  return p + 1;
}

// CHECK-MESSAGES: :[[@LINE+1]]:27: warning: pointer parameter 'p' can be
const char *return4(char *p) {
  // CHECK-FIXES: const char *return4(const char *p) {
  return p ? p : "";
}

char *return5(char *s) {
  return s;
}

char *return6(char *s) {
  return s + 1;
}

char *return7(char *a, char *b) {
  return a ? a : b;
}

char return8(int *p) {
  return ++(*p);
}

void dontwarn1(int *p) {
  ++(*p);
}

void dontwarn2(int *p) {
  (*p)++;
}

int dontwarn3(_Atomic(int) * p) {
  return *p;
}

void callFunction1(char *p) {
  strcpy1(p, "abc");
}

void callFunction2(char *p) {
  strcpy1(&p[0], "abc");
}

void callFunction3(char *p) {
  strcpy1(p + 2, "abc");
}

char *callFunction4(char *p) {
  return strcpy1(p, "abc");
}

unsigned callFunction5(char *buf) {
  unsigned len = my_strlen(buf);
  return len + my_strcpy(buf, "abc");
}

void f6(int **p);
void callFunction6(int *p) { f6(&p); }

typedef union { void *v; } t;
void f7(t obj);
void callFunction7(int *p) {
  f7((t){p});
}

void f8(int &x);
void callFunction8(int *p) {
  f8(*p);
}

// Don't warn about nonconst function pointers that can be const.
void functionpointer(double f(double), int x) {
  f(x);
}

// TODO: This is a false positive.
// CHECK-MESSAGES: :[[@LINE+1]]:27: warning: pointer parameter 'p' can be
int functionpointer2(int *p) {
  return *p;
}
void use_functionpointer2() {
  int (*fp)(int *) = functionpointer2; // <- the parameter 'p' can't be const
}

// Don't warn about nonconst record pointers that can be const.
struct XY {
  int *x;
  int *y;
};
void recordpointer(struct XY *xy) {
  *(xy->x) = 0;
}
void recordInitList(int *x) {
  XY xy = {x, nullptr};
}

struct XYConst {
  int const *x;
};
// CHECK-MESSAGES: :[[@LINE+1]]:30: warning: pointer parameter 'x' can be pointer to const
void recordInitListDiag(int *x) {
  // CHECK-FIXES: void recordInitListDiag(const int *x) {
  XYConst xy = {x};
}
typedef XYConst XYConstAlias;
// CHECK-MESSAGES: :[[@LINE+1]]:35: warning: pointer parameter 'x' can be pointer to const
void recordInitListAliasDiag(int *x) {
  // CHECK-FIXES: void recordInitListAliasDiag(const int *x) {
  XYConstAlias xy = {x};
}

class C {
public:
  C(int *p) : p(p) {}

private:
  int *p;
};

class C2 {
public:
  // CHECK-MESSAGES: :[[@LINE+1]]:11: warning: pointer parameter 'p' can be
  C2(int *p) : p(p) {}
  // CHECK-FIXES: C2(const int *p) : p(p) {}

private:
  const int *p;
};

void tempObject(int *p) {
  C c(p);
}

// avoid fp for const pointer array
void constPointerArray(const char *remapped[][2]) {
  const char *name = remapped[0][0];
}

class Warn {
public:
  // CHECK-MESSAGES: :[[@LINE+1]]:21: warning: pointer parameter 'p' can be
  void doStuff(int *p) {
    // CHECK-FIXES: void doStuff(const int *p) {
    x = *p;
  }

private:
  int x;
};

class Base {
public:
  // Ensure there is no false positive for this method. It is virtual.
  virtual void doStuff(int *p) {
    int x = *p;
  }
};

class Derived : public Base {
public:
  // Ensure there is no false positive for this method. It overrides a method.
  void doStuff(int *p) override {
    int x = *p;
  }
};

extern char foo(char *s); // 1
// CHECK-FIXES: extern char foo(const char *s); // 1
// CHECK-MESSAGES: :[[@LINE+1]]:16: warning: pointer parameter 's' can be
char foo(char *s) {
  // CHECK-FIXES: char foo(const char *s) {
  return *s;
}
char foo(char *s); // 2
// CHECK-FIXES: char foo(const char *s); // 2

void lvalueReference(int *p) {
  // CHECK-MESSAGES-NOT: warning: pointer parameter 'p' can be
  int &x = *p;
}

// CHECK-MESSAGES: :[[@LINE+1]]:32: warning: pointer parameter 'p' can be
void constLValueReference(int *p) {
  // CHECK-FIXES: void constLValueReference(const int *p) {
  const int &x = *p;
}

void lambdaLVRef(int *p) {
  // CHECK-MESSAGES-NOT: warning: pointer parameter 'p' can be
  auto foo = [&]() {
    int &x = *p;
  };
}

// CHECK-MESSAGES: :[[@LINE+1]]:28: warning: pointer parameter 'p' can be
void lambdaConstLVRef(int *p) {
  // CHECK-FIXES: void lambdaConstLVRef(const int *p) {
  auto foo = [&]() {
    const int &x = *p;
  };
}

struct Temp1 {
  Temp1(int &i) {
    i = 10;
  }
};
void constructLVRef(int *p) {
  // CHECK-MESSAGES-NOT: warning: pointer parameter 'p' can be
  Temp1 t(*p);
}
