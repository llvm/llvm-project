// RUN: %clang_cc1 -triple x86_64-linux -verify=expected,all                   -std=c11 -Wcast-qual           %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -triple x86_64-linux -verify=pedantic,pedantic-expected,all -std=c11 -Wcast-qual -pedantic %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -triple x86_64-linux -verify=ref,all                        -std=c11 -Wcast-qual           %s
// RUN: %clang_cc1 -triple x86_64-linux -verify=pedantic,pedantic-ref,all      -std=c11 -Wcast-qual -pedantic %s

typedef __INTPTR_TYPE__ intptr_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;

_Static_assert(1, "");

_Static_assert(__objc_yes, "");
_Static_assert(!__objc_no, "");

_Static_assert(0 != 1, "");
_Static_assert(1.0 == 1.0, ""); // pedantic-ref-warning {{not an integer constant expression}} \
                                // pedantic-expected-warning {{not an integer constant expression}}
_Static_assert(1 && 1.0, ""); // pedantic-ref-warning {{not an integer constant expression}} \
                              // pedantic-expected-warning {{not an integer constant expression}}
_Static_assert( (5 > 4) + (3 > 2) == 2, "");
_Static_assert(!!1.0, ""); // pedantic-ref-warning {{not an integer constant expression}} \
                           // pedantic-expected-warning {{not an integer constant expression}}
_Static_assert(!!1, "");

_Static_assert(!(_Bool){(void*)0}, ""); // pedantic-ref-warning {{not an integer constant expression}} \
                                        // pedantic-expected-warning {{not an integer constant expression}}

int a = (1 == 1 ? 5 : 3);
_Static_assert(a == 5, ""); // all-error {{not an integral constant expression}}

const int DiscardedPtrToIntCast = ((intptr_t)((void*)0), 0); // all-warning {{left operand of comma operator has no effect}}

const int b = 3;
_Static_assert(b == 3, ""); // pedantic-ref-warning {{not an integer constant expression}} \
                            // pedantic-expected-warning {{not an integer constant expression}}

const int c; // all-note {{declared here}}
_Static_assert(c == 0, ""); // ref-error {{not an integral constant expression}} \
                            // ref-note {{initializer of 'c' is unknown}} \
                            // pedantic-ref-error {{not an integral constant expression}} \
                            // pedantic-ref-note {{initializer of 'c' is unknown}} \
                            // expected-error {{not an integral constant expression}} \
                            // expected-note {{initializer of 'c' is unknown}} \
                            // pedantic-expected-error {{not an integral constant expression}} \
                            // pedantic-expected-note {{initializer of 'c' is unknown}}

_Static_assert(&c != 0, ""); // ref-warning {{always true}} \
                             // pedantic-ref-warning {{always true}} \
                             // pedantic-ref-warning {{is a GNU extension}} \
                             // expected-warning {{always true}} \
                             // pedantic-expected-warning {{always true}} \
                             // pedantic-expected-warning {{is a GNU extension}}
_Static_assert(&a != 0, ""); // ref-warning {{always true}} \
                             // pedantic-ref-warning {{always true}} \
                             // pedantic-ref-warning {{is a GNU extension}} \
                             // expected-warning {{always true}} \
                             // pedantic-expected-warning {{always true}} \
                             // pedantic-expected-warning {{is a GNU extension}}
_Static_assert((&c + 1) != 0, ""); // pedantic-ref-warning {{is a GNU extension}} \
                                   // pedantic-expected-warning {{is a GNU extension}}
_Static_assert((&a + 100) != 0, ""); // pedantic-ref-warning {{is a GNU extension}} \
                                     // pedantic-ref-note {{100 of non-array}} \
                                     // pedantic-expected-note {{100 of non-array}} \
                                     // pedantic-expected-warning {{is a GNU extension}}
_Static_assert((&a - 100) != 0, ""); // pedantic-ref-warning {{is a GNU extension}} \
                                     // pedantic-expected-warning {{is a GNU extension}} \
                                     // pedantic-ref-note {{-100 of non-array}} \
                                     // pedantic-expected-note {{-100 of non-array}}
/// extern variable of a composite type.
extern struct Test50S Test50;
_Static_assert(&Test50 != (void*)0, ""); // all-warning {{always true}} \
                                         // pedantic-warning {{is a GNU extension}} \
                                         // pedantic-note {{this conversion is not allowed in a constant expression}}

struct y {int x,y;};
int a2[(intptr_t)&((struct y*)0)->y]; // all-warning {{folded to constant array}}

const struct y *yy = (struct y*)0;
const intptr_t L = (intptr_t)(&(yy->y)); // all-error {{not a compile-time constant}}

_Static_assert((long)&((struct y*)0)->y > 0, ""); // pedantic-ref-warning {{GNU extension}} \
                                                  // pedantic-ref-note {{this conversion is not allowed in a constant expression}} \
                                                  // pedantic-expected-warning {{GNU extension}} \
                                                  // pedantic-expected-note {{this conversion is not allowed in a constant expression}}

const ptrdiff_t m = &m + 137 - &m;
_Static_assert(m == 137, ""); // pedantic-ref-warning {{GNU extension}} \
                              // pedantic-expected-warning {{GNU extension}}

/// from test/Sema/switch.c, used to cause an assertion failure.
void f (int z) {
  while (z) {
    default: z--; // all-error {{'default' statement not in switch}}
  }
}

int expr;
int chooseexpr[__builtin_choose_expr(1, 1, expr)];

int somefunc(int i) {
  return (i, 65537) * 65537; // all-warning {{left operand of comma operator has no effect}} \
                             // all-warning {{overflow in expression; result is 131'073 with type 'int'}}
}

#pragma clang diagnostic ignored "-Wpointer-to-int-cast"
struct ArrayStruct {
  char n[1];
};
char name2[(int)&((struct ArrayStruct*)0)->n]; // expected-warning {{folded to constant array}} \
                                               // pedantic-expected-warning {{folded to constant array}} \
                                               // ref-warning {{folded to constant array}} \
                                               // pedantic-ref-warning {{folded to constant array}}
_Static_assert(sizeof(name2) == 0, "");

#ifdef __SIZEOF_INT128__
void *PR28739d = &(&PR28739d)[(__int128)(unsigned long)-1]; // all-warning {{refers past the last possible element}}
#endif

extern float global_float;
struct XX { int a, *b; };
struct XY { int before; struct XX xx, *xp; float* after; } xy[] = {
  0, 0, &xy[0].xx.a, &xy[0].xx, &global_float,
  [1].xx = 0, &xy[1].xx.a, &xy[1].xx, &global_float,
  0,              // all-note {{previous initialization is here}}
  0,              // all-note {{previous initialization is here}}
  [2].before = 0, // all-warning {{initializer overrides prior initialization of this subobject}}
  0,              // all-warning {{initializer overrides prior initialization of this subobject}}
  &xy[2].xx.a, &xy[2].xx, &global_float
};

void t14(void) {
  int array[256] = { 0 }; // expected-note {{array 'array' declared here}} \
                          // pedantic-expected-note {{array 'array' declared here}} \
                          // ref-note {{array 'array' declared here}} \
                          // pedantic-ref-note {{array 'array' declared here}}
  const char b = -1;
  int val = array[b]; // expected-warning {{array index -1 is before the beginning of the array}} \
                      // pedantic-expected-warning {{array index -1 is before the beginning of the array}} \
                      // ref-warning {{array index -1 is before the beginning of the array}} \
                      // pedantic-ref-warning {{array index -1 is before the beginning of the array}}

}

void bar_0(void) {
  struct C {
    const int a;
    int b;
  };

  const struct C S = {0, 0};

  *(int *)(&S.a) = 0; // all-warning {{cast from 'const int *' to 'int *' drops const qualifier}}
  *(int *)(&S.b) = 0; // all-warning {{cast from 'const int *' to 'int *' drops const qualifier}}
}

/// Complex-to-bool casts.
const int A =  ((_Complex double)1.0 ? 21 : 1);
_Static_assert(A == 21, ""); // pedantic-ref-warning {{GNU extension}} \
                             // pedantic-expected-warning {{GNU extension}}

const int CTI1 = ((_Complex double){0.0, 1.0}); // pedantic-ref-warning {{extension}} \
                                                // pedantic-expected-warning {{extension}}
_Static_assert(CTI1 == 0, ""); // pedantic-ref-warning {{GNU extension}} \
                               // pedantic-expected-warning {{GNU extension}}

const _Bool CTB2 = (_Bool)(_Complex double){0.0, 1.0}; // pedantic-ref-warning {{extension}} \
                                                       // pedantic-expected-warning {{extension}}
_Static_assert(CTB2, ""); // pedantic-ref-warning {{GNU extension}} \
                          // pedantic-expected-warning {{GNU extension}}

const _Bool CTB3 = (_Complex double){0.0, 1.0}; // pedantic-ref-warning {{extension}} \
                                                // pedantic-expected-warning {{extension}}
_Static_assert(CTB3, ""); // pedantic-ref-warning {{GNU extension}} \
                          // pedantic-expected-warning {{GNU extension}}


void nonComplexToComplexCast(void) {
  _Complex double z = *(_Complex double *)&(struct { double r, i; }){0.0, 1.0};
}

int t1 = sizeof(int);
void test4(void) {
  t1 = sizeof(int);
}

void localCompoundLiteral(void) {
  struct S { int x, y; } s = {}; // pedantic-expected-warning {{use of an empty initializer}} \
                                 // pedantic-ref-warning {{use of an empty initializer}}
  struct T {
	int i;
    struct S s;
  } t1 = { 1, {} }; // pedantic-expected-warning {{use of an empty initializer}} \
                    // pedantic-ref-warning {{use of an empty initializer}}

  struct T t3 = {
    (int){}, // pedantic-expected-warning {{use of an empty initializer}} \
             // pedantic-ref-warning {{use of an empty initializer}}
    {} // pedantic-expected-warning {{use of an empty initializer}} \
       // pedantic-ref-warning {{use of an empty initializer}}
  };
}

/// struct copy
struct StrA {int a; };
const struct StrA sa = { 12 };
const struct StrA * const sb = &sa;
const struct StrA sc = *sb;
_Static_assert(sc.a == 12, ""); // pedantic-ref-warning {{GNU extension}} \
                                // pedantic-expected-warning {{GNU extension}}

struct ComplexS {
  int a;
  float b;
  struct StrA sa[2];
};
const struct ComplexS CS = {12, 23.0f, {{1}, {2}}};
const struct ComplexS CS2 = CS;
_Static_assert(CS2.sa[0].a == 1, ""); // pedantic-ref-warning {{GNU extension}} \
                                      // pedantic-expected-warning {{GNU extension}}

_Static_assert(((void*)0 + 1) != (void*)0, ""); // pedantic-expected-warning {{arithmetic on a pointer to void is a GNU extension}} \
                                                // pedantic-expected-warning {{not an integer constant expression}} \
                                                // pedantic-expected-note {{cannot perform pointer arithmetic on null pointer}} \
                                                // pedantic-ref-warning {{arithmetic on a pointer to void is a GNU extension}} \
                                                // pedantic-ref-warning {{not an integer constant expression}} \
                                                // pedantic-ref-note {{cannot perform pointer arithmetic on null pointer}}

typedef __INTPTR_TYPE__ intptr_t;
int array[(intptr_t)(int*)1]; // ref-warning {{variable length array folded to constant array}} \
                              // pedantic-ref-warning {{variable length array folded to constant array}} \
                              // expected-warning {{variable length array folded to constant array}} \
                              // pedantic-expected-warning {{variable length array folded to constant array}}

int castViaInt[*(int*)(unsigned long)"test"]; // ref-error {{variable length array}} \
                                              // pedantic-ref-error {{variable length array}} \
                                              // expected-error {{variable length array}} \
                                              // pedantic-expected-error {{variable length array}}

const void (*const funcp)(void) = (void*)123; // pedantic-warning {{converts between void pointer and function pointer}} \
                                              // pedantic-expected-note {{this conversion is not allowed in a constant expression}}
_Static_assert(funcp == (void*)0, ""); // all-error {{failed due to requirement 'funcp == (void *)0'}} \
                                       // pedantic-warning {{expression is not an integer constant expression}}
_Static_assert(funcp == (void*)123, ""); // pedantic-warning {{equality comparison between function pointer and void pointer}} \
                                         // pedantic-warning {{expression is not an integer constant expression}}

void unaryops(void) {
  (void)(++(struct x {unsigned x;}){3}.x);
  (void)(--(struct y {unsigned x;}){3}.x);
  (void)(++(struct z {float x;}){3}.x);
  (void)(--(struct w {float x;}){3}.x);

  (void)((struct xx {unsigned x;}){3}.x++);
  (void)((struct yy {unsigned x;}){3}.x--);
  (void)((struct zz {float x;}){3}.x++);
  (void)((struct ww {float x;}){3}.x--);
}

/// This used to fail because we didn't properly mark the struct
/// initialized through a CompoundLiteralExpr as initialized.
struct TestStruct {
  int a;
  int b;
};
int Y __attribute__((annotate(
  "GlobalValAnnotationWithArgs",
  42,
  (struct TestStruct) { .a = 1, .b = 2 }
)));

#ifdef __SIZEOF_INT128__
const int *p = &b;
const __int128 K = (__int128)(int*)0;
const unsigned __int128 KU = (unsigned __int128)(int*)0;
#endif


int test3(void) {
  int a[2];
  a[0] = test3; // all-error {{incompatible pointer to integer conversion assigning to 'int' from 'int (void)'}}
  return 0;
}
/// This tests that we have full type info, even for values we cannot read.
int dummyarray[5];
_Static_assert(&dummyarray[0] < &dummyarray[1], ""); // pedantic-warning {{GNU extension}}

void addrlabelexpr(void) {
 a0: ;
  static void *ps[] = { &&a0 }; // pedantic-warning {{use of GNU address-of-label extension}}
}

extern void cv2;
void *foo5 (void)
{
  return &cv2; // pedantic-warning{{address of an expression of type 'void'}}
}

__attribute__((weak)) const unsigned int test10_bound = 10;
char test10_global[test10_bound]; // all-error {{variable length array declaration not allowed at file scope}}
void test10(void) {
  char test10_local[test10_bound] = "help"; // all-error {{variable-sized object may not be initialized}}
}

void SuperSpecialFunc(void) {
const int SuperSpecialCase = 10;
_Static_assert((sizeof(SuperSpecialCase) == 12 && SuperSpecialCase == 3) || SuperSpecialCase == 10, ""); // pedantic-warning {{GNU extension}}
}


void T1(void) {
  static int *y[1] = {({ static int _x = 20; (void*)0;})}; // all-error {{initializer element is not a compile-time constant}} \
                                                           // pedantic-warning {{use of GNU statement expression extension}}
}

enum teste1 test1f(void), (*test1)(void) = test1f; // pedantic-warning {{ISO C forbids forward references to 'enum' types}}
enum teste1 { TEST1 };

void func(void) {
  _Static_assert(func + 1 - func == 1, ""); // pedantic-warning {{arithmetic on a pointer to the function type}} \
                                            // pedantic-warning {{arithmetic on pointers to the function type}} \
                                            // pedantic-warning {{not an integer constant expression}}
  _Static_assert(func + 0xdead000000000000UL - 0xdead000000000000UL == func, ""); // pedantic-warning 2{{arithmetic on a pointer to the function type}} \
                                                                                  // pedantic-warning {{not an integer constant expression}} \
                                                                                  // pedantic-note {{cannot refer to element 16045481047390945280 of non-array object in a constant expression}}
  _Static_assert(func + 1 != func, ""); // pedantic-warning {{arithmetic on a pointer to the function type}} \
                                        // pedantic-warning {{expression is not an integer constant expression}}
  func + 0xdead000000000000UL; // all-warning {{expression result unused}} \
                               // pedantic-warning {{arithmetic on a pointer to the function type}}
  func - 0xdead000000000000UL; // all-warning {{expression result unused}} \
                               // pedantic-warning {{arithmetic on a pointer to the function type}}
}

void foo3 (void)
{
 void* x = 0;
 void* y = &*x;
}

static void *FooTable[1] = {
    [0] = (void *[1]) { // 1
        [0] = (void *[1]) { // 2
            [0] = (void *[1]) {} // pedantic-warning {{use of an empty initializer}}
        },
    }
};

int strcmp(const char *, const char *); // all-note {{passing argument to parameter here}}
#define S "\x01\x02\x03\x04\x05\x06\x07\x08"
const char _str[] = {S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7]};
const unsigned char _str2[] = {S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7]};
const int compared = strcmp(_str, (const char *)_str2); // all-error {{initializer element is not a compile-time constant}}


const int compared2 = strcmp(strcmp, _str); // all-warning {{incompatible pointer types}} \
                                            // all-error {{initializer element is not a compile-time constant}}

int foo(x) // all-warning {{a function definition without a prototype is deprecated in all versions of C}}
int x;
{
  return x;
}

void bar() { // pedantic-warning {{a function declaration without a prototype}}
  int x;
  x = foo(); // all-warning {{too few arguments}}
}
