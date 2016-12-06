// RUN: %clang_cc1 -triple i386-apple-darwin10 -analyze -analyzer-checker=core,deadcode,alpha.core -std=gnu99 -analyzer-store=region -analyzer-purge=none -verify %s -Wno-error=return-type
// RUN: %clang_cc1 -triple i386-apple-darwin10 -analyze -analyzer-checker=core,deadcode,alpha.core -std=gnu99 -analyzer-store=region -verify %s -Wno-error=return-type

typedef unsigned uintptr_t;

extern void __assert_fail (__const char *__assertion, __const char *__file,
    unsigned int __line, __const char *__function)
     __attribute__ ((__noreturn__));

#define assert(expr) \
  ((expr)  ? (void)(0)  : __assert_fail (#expr, __FILE__, __LINE__, __func__))

void f1(int *p) {  
  if (p) *p = 1;
  else *p = 0; // expected-warning{{ereference}}
}

struct foo_struct {
  int x;
};

int f2(struct foo_struct* p) {
  
  if (p)
    p->x = 1;
    
  return p->x++; // expected-warning{{Access to field 'x' results in a dereference of a null pointer (loaded from variable 'p')}}
}

int f3(char* x) {
  
  int i = 2;
  
  if (x)
    return x[i - 1];
  
  return x[i+1]; // expected-warning{{Array access (from variable 'x') results in a null pointer dereference}}
}

int f3_b(char* x) {
  
  int i = 2;
  
  if (x)
    return x[i - 1];
  
  return x[i+1]++; // expected-warning{{Array access (from variable 'x') results in a null pointer dereference}}
}

int f4(int *p) {
  
  uintptr_t x = (uintptr_t) p;
  
  if (x)
    return 1;
    
  int *q = (int*) x;
  return *q; // expected-warning{{Dereference of null pointer (loaded from variable 'q')}}
}

int f4_b() {
  short array[2];
  uintptr_t x = array; // expected-warning{{incompatible pointer to integer conversion}}
  short *p = x; // expected-warning{{incompatible integer to pointer conversion}}

  // The following branch should be infeasible.
  if (!(p == &array[0])) {
    p = 0;
    *p = 1; // no-warning
  }

  if (p) {
    *p = 5; // no-warning
    p = 0;
  }
  else return; // expected-warning {{non-void function 'f4_b' should return a value}}

  *p += 10; // expected-warning{{Dereference of null pointer}}
  return 0;
}

int f5() {
  
  char *s = "hello world";
  return s[0]; // no-warning
}

int bar(int* p, int q) __attribute__((nonnull));

int f6(int *p) { 
  return !p ? bar(p, 1) // expected-warning {{Null pointer passed as an argument to a 'nonnull' parameter}}
         : bar(p, 0);   // no-warning
}

int bar2(int* p, int q) __attribute__((nonnull(1)));

int f6b(int *p) { 
  return !p ? bar2(p, 1) // expected-warning {{Null pointer passed as an argument to a 'nonnull' parameter}}
         : bar2(p, 0);   // no-warning
}

int bar3(int*p, int q, int *r) __attribute__((nonnull(1,3)));

int f6c(int *p, int *q) {
   return !p ? bar3(q, 2, p) // expected-warning {{Null pointer passed as an argument to a 'nonnull' parameter}}
             : bar3(p, 2, q); // no-warning
}

void f6d(int *p) {
  bar(p, 0);
  // At this point, 'p' cannot be null.
  if (!p) {
    int *q = 0;
    *q = 0xDEADBEEF; // no-warning    
  }  
}

void f6e(int *p, int offset) {
  // PR7406 - crash from treating an UnknownVal as defined, to see if it's 0.
  bar((p+offset)+1, 0); // not crash
}

int* qux();

int f7(int x) {
  
  int* p = 0;
  
  if (0 == x)
    p = qux();
  
  if (0 == x)
    *p = 1; // no-warning
    
  return x;
}

int* f7b(int *x) {
  
  int* p = 0;
  
  if (((void*)0) == x)
    p = qux();
  
  if (((void*)0) == x)
    *p = 1; // no-warning
    
  return x;
}

int* f7c(int *x) {
  
  int* p = 0;
  
  if (((void*)0) == x)
    p = qux();
  
  if (((void*)0) != x)
    return x;

  // If we reach here then 'p' is not null.
  *p = 1; // no-warning
  return x;
}

int* f7c2(int *x) {
  
  int* p = 0;
  
  if (((void*)0) == x)
    p = qux();
  
  if (((void*)0) == x)
    return x;
    
  *p = 1; // expected-warning{{null}}
  return x;
}


void f8(int *p, int *q) {
  if (!p)
    if (p)
      *p = 1; // no-warning
  
  if (q)
    if (!q)
      *q = 1; // no-warning
}

int* qux();

int f9(unsigned len) {
  assert (len != 0);
  int *p = 0;
  unsigned i;

  for (i = 0; i < len; ++i)
   p = qux(i);

  return *p++; // no-warning
}

int f9b(unsigned len) {
  assert (len > 0);  // note use of '>'
  int *p = 0;
  unsigned i;

  for (i = 0; i < len; ++i)
   p = qux(i);

  return *p++; // no-warning
}

int* f10(int* p, signed char x, int y) {
  // This line tests symbolication with compound assignments where the
  // LHS and RHS have different bitwidths.  The new symbolic value
  // for 'x' should have a bitwidth of 8.
  x &= y;
  
  // This tests that our symbolication worked, and that we correctly test
  // x against 0 (with the same bitwidth).
  if (!x) {
    if (!p) return 0;
    *p = 10;
  }
  else p = 0;

  if (!x)
    *p = 5; // no-warning

  return p;
}

// Test case from <rdar://problem/6407949>
void f11(unsigned i) {
  int *x = 0;
  if (i >= 0) { // expected-warning{{always true}}
    // always true
  } else {
    *x = 42; // no-warning
  }
}

void f11b(unsigned i) {
  int *x = 0;
  if (i <= ~(unsigned)0) {
    // always true
  } else {
    *x = 42; // no-warning
  }
}

// Test case for switch statements with weird case arms.
typedef int     BOOL, *PBOOL, *LPBOOL;
typedef long    LONG_PTR, *PLONG_PTR;
typedef unsigned long ULONG_PTR, *PULONG_PTR;
typedef ULONG_PTR DWORD_PTR, *PDWORD_PTR;
typedef LONG_PTR LRESULT;
typedef struct _F12ITEM *HF12ITEM;

void f12(HF12ITEM i, char *q) {
  char *p = 0;
  switch ((DWORD_PTR) i) {
  case 0 ... 10:
    p = q;
    break;
  case (DWORD_PTR) ((HF12ITEM) - 65535):
    return;
  default:
    return;
  }
  
  *p = 1; // no-warning
}

// Test handling of translating between integer "pointers" and back.
void f13() {
  int *x = 0;
  if (((((int) x) << 2) + 1) >> 1) *x = 1;
}

// PR 4759 - Attribute non-null checking by the analyzer was not correctly
// handling pointer values that were undefined.
void pr4759_aux(int *p) __attribute__((nonnull));

void pr4759() {
  int *p;
  pr4759_aux(p); // expected-warning{{Function call argument is an uninitialized value}}
}

// Relax function call arguments invalidation to be aware of const
// arguments. Test with function pointers. radar://10595327
void ttt(const int *nptr);
void ttt2(const int *nptr);
typedef void (*NoConstType)(int*);
int foo10595327(int b) {
  void (*fp)(int *);
  // We use path sensitivity to get the function declaration. Even when the
  // function pointer is cast to non-pointer-to-const parameter type, we can
  // find the right function declaration.
  if (b > 5)
    fp = (NoConstType)ttt2;
  else
    fp = (NoConstType)ttt;
  int x = 3;
  int y = x + 1;
  int *p = 0;
  fp(&y);
  if (x == y)
      return *p; // no-warning
  return 0;
}

#define AS_ATTRIBUTE volatile __attribute__((address_space(256)))
#define _get_base() ((void * AS_ATTRIBUTE *)0)
void* test_address_space_array(unsigned long slot) {
  return _get_base()[slot]; // no-warning
}
void test_address_space_condition(int AS_ATTRIBUTE *cpu_data) {
   if (cpu_data == 0) {
    *cpu_data = 3; // no-warning
  }
}
struct X { int member; };
int test_address_space_member() {
  struct X AS_ATTRIBUTE *data = (struct X AS_ATTRIBUTE *)0UL;
  int ret;
  ret = data->member; // no-warning
  return ret;
}
