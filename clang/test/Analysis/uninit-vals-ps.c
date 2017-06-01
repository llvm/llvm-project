// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-store=region -fblocks -verify %s

struct FPRec {
  void (*my_func)(int * x);  
};

int bar(int x);

int f1_a(struct FPRec* foo) {
  int x;
  (*foo->my_func)(&x);
  return bar(x)+1; // no-warning
}

int f1_b() {
  int x;
  return bar(x)+1;  // expected-warning{{1st function call argument is an uninitialized value}}
}

int f2() {
  
  int x;
  
  if (x+1)  // expected-warning{{The left operand of '+' is a garbage value}}
    return 1;
    
  return 2;  
}

int f2_b() {
  int x;
  
  return ((1+x)+2+((x))) + 1 ? 1 : 2; // expected-warning{{The right operand of '+' is a garbage value}}
}

int f3(void) {
  int i;
  int *p = &i;
  if (*p > 0) // expected-warning{{The left operand of '>' is a garbage value}}
    return 0;
  else
    return 1;
}

void f4_aux(float* x);
float f4(void) {
  float x;
  f4_aux(&x);
  return x;  // no-warning
}

struct f5_struct { int x; };
void f5_aux(struct f5_struct* s);
int f5(void) {
  struct f5_struct s;
  f5_aux(&s);
  return s.x; // no-warning
}

void f6(int x) {
  int a[20];
  if (x == 25) {}
  if (a[x] == 123) {} // expected-warning{{The left operand of '==' is a garbage value due to array index out of bounds}}
}

int ret_uninit() {
  int i;
  int *p = &i;
  return *p;  // expected-warning{{Undefined or garbage value returned to caller}}
}

// <rdar://problem/6451816>
typedef unsigned char Boolean;
typedef const struct __CFNumber * CFNumberRef;
typedef signed long CFIndex;
typedef CFIndex CFNumberType;
typedef unsigned long UInt32;
typedef UInt32 CFStringEncoding;
typedef const struct __CFString * CFStringRef;
extern Boolean CFNumberGetValue(CFNumberRef number, CFNumberType theType, void *valuePtr);
extern CFStringRef CFStringConvertEncodingToIANACharSetName(CFStringEncoding encoding);

CFStringRef rdar_6451816(CFNumberRef nr) {
  CFStringEncoding encoding;
  // &encoding is casted to void*.  This test case tests whether or not
  // we properly invalidate the value of 'encoding'.
  CFNumberGetValue(nr, 9, &encoding);
  return CFStringConvertEncodingToIANACharSetName(encoding); // no-warning
}

// PR 4630 - false warning with nonnull attribute
//  This false positive (due to a regression) caused the analyzer to falsely
//  flag a "return of uninitialized value" warning in the first branch due to
//  the nonnull attribute.
void pr_4630_aux(char *x, int *y) __attribute__ ((nonnull (1)));
void pr_4630_aux_2(char *x, int *y);
int pr_4630(char *a, int y) {
  int x;
  if (y) {
    pr_4630_aux(a, &x);
    return x;   // no-warning
  }
  else {
    pr_4630_aux_2(a, &x);
    return x;   // no-warning
  }
}

// PR 4631 - False positive with union initializer
//  Previously the analyzer didn't examine the compound initializers of unions,
//  resulting in some false positives for initializers with side-effects.
union u_4631 { int a; };
struct s_4631 { int a; };
int pr4631_f2(int *p);
int pr4631_f3(void *q);
int pr4631_f1(void)
{
  int x;
  union u_4631 m = { pr4631_f2(&x) };
  pr4631_f3(&m); // tell analyzer that we use m
  return x;  // no-warning
}
int pr4631_f1_b(void)
{
  int x;
  struct s_4631 m = { pr4631_f2(&x) };
  pr4631_f3(&m); // tell analyzer that we use m
  return x;  // no-warning
}

// <rdar://problem/12278788> - FP when returning a void-valued expression from
// a void function...or block.
void foo_radar12278788() { return; }
void test_radar12278788() {
  return foo_radar12278788(); // no-warning
}

void foo_radar12278788_fp() { return; }
typedef int (*RetIntFuncType)();
typedef void (*RetVoidFuncType)();
int test_radar12278788_FP() {
  RetVoidFuncType f = foo_radar12278788_fp;
  return ((RetIntFuncType)f)(); //expected-warning {{Undefined or garbage value returned to caller}}
}

void rdar13665798() {
  ^() {
    return foo_radar12278788(); // no-warning
  }();
  ^void() {
    return foo_radar12278788(); // no-warning
  }();
  ^int() {
    RetVoidFuncType f = foo_radar12278788_fp;
    return ((RetIntFuncType)f)(); //expected-warning {{Undefined or garbage value returned to caller}}
  }();
}
