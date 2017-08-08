// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-linux %s -Wno-tautological-pointer-compare

#define EVAL_EXPR(testno, expr) int test##testno = sizeof(struct{char qq[expr];});
int x;
EVAL_EXPR(1, (_Bool)&x)
EVAL_EXPR(2, (int)(1.0+(double)4))
EVAL_EXPR(3, (int)(1.0+(float)4.0))
EVAL_EXPR(4, (_Bool)(1 ? (void*)&x : 0))
EVAL_EXPR(5, (_Bool)(int[]){0})
struct y {int x,y;};
EVAL_EXPR(6, (int)(1+(struct y*)0))
EVAL_EXPR(7, (int)&((struct y*)0)->y)
EVAL_EXPR(8, (_Bool)"asdf")
EVAL_EXPR(9, !!&x)
EVAL_EXPR(10, ((void)1, 12))
void g0(void);
EVAL_EXPR(11, (g0(), 12)) // expected-error {{must have a constant size}}
EVAL_EXPR(12, 1.0&&2.0)
EVAL_EXPR(13, x || 3.0) // expected-error {{must have a constant size}}

unsigned int l_19 = 1;
EVAL_EXPR(14, (1 ^ l_19) && 1); // expected-error {{fields must have a constant size}}

void f()
{
  int a;
  EVAL_EXPR(15, (_Bool)&a);
}

// FIXME: Turn into EVAL_EXPR test once we have more folding.
_Complex float g16 = (1.0f + 1.0fi);

// ?: in constant expressions.
int g17[(3?:1) - 2]; 

EVAL_EXPR(18, ((int)((void*)10 + 10)) == 20 ? 1 : -1);

struct s {
  int a[(int)-1.0f]; // expected-error {{'a' declared as an array with a negative size}}
};

EVAL_EXPR(19, ((int)&*(char*)10 == 10 ? 1 : -1));

EVAL_EXPR(20, __builtin_constant_p(*((int*) 10)));

EVAL_EXPR(21, (__imag__ 2i) == 2 ? 1 : -1);

EVAL_EXPR(22, (__real__ (2i+3)) == 3 ? 1 : -1);

int g23[(int)(1.0 / 1.0)] = { 1 };
int g24[(int)(1.0 / 1.0)] = { 1 , 2 }; // expected-warning {{excess elements in array initializer}}
int g25[(int)(1.0 + 1.0)], g26 = sizeof(g25);

EVAL_EXPR(26, (_Complex double)0 ? -1 : 1)
EVAL_EXPR(27, (_Complex int)0 ? -1 : 1)
EVAL_EXPR(28, (_Complex double)1 ? 1 : -1)
EVAL_EXPR(29, (_Complex int)1 ? 1 : -1)


// PR4027 + rdar://6808859
struct a { int x, y; };
static struct a V2 = (struct a)(struct a){ 1, 2};
static const struct a V1 = (struct a){ 1, 2};

EVAL_EXPR(30, (int)(_Complex float)((1<<30)-1) == (1<<30) ? 1 : -1)
EVAL_EXPR(31, (int*)0 == (int*)0 ? 1 : -1)
EVAL_EXPR(32, (int*)0 != (int*)0 ? -1 : 1)
EVAL_EXPR(33, (void*)0 - (void*)0 == 0 ? 1 : -1)
void foo(void) {}
EVAL_EXPR(34, (foo == (void *)0) ? -1 : 1)

// No PR. Mismatched bitwidths lead to a crash on second evaluation.
const _Bool constbool = 0;
EVAL_EXPR(35, constbool)
EVAL_EXPR(36, constbool)

EVAL_EXPR(37, (1,2.0) == 2.0 ? 1 : -1)
EVAL_EXPR(38, __builtin_expect(1,1) == 1 ? 1 : -1)

// PR7884
EVAL_EXPR(39, __real__(1.f) == 1 ? 1 : -1)
EVAL_EXPR(40, __imag__(1.f) == 0 ? 1 : -1)

// From gcc testsuite
EVAL_EXPR(41, (int)(1+(_Complex unsigned)2))

// rdar://8875946
void rdar8875946() {
  double _Complex  P;
  float _Complex  P2 = 3.3f + P;
}

double d = (d = 0.0); // expected-error {{not a compile-time constant}}
double d2 = ++d; // expected-error {{not a compile-time constant}}

int n = 2;
int intLvalue[*(int*)((long)&n ?: 1)] = { 1, 2 }; // expected-error {{variable length array}}

union u { int a; char b[4]; };
char c = ((union u)(123456)).b[0]; // expected-error {{not a compile-time constant}}

extern const int weak_int __attribute__((weak));
const int weak_int = 42;
int weak_int_test = weak_int; // expected-error {{not a compile-time constant}}

int literalVsNull1 = "foo" == 0;
int literalVsNull2 = 0 == "foo";

// PR11385.
int castViaInt[*(int*)(unsigned long)"test"]; // expected-error {{variable length array}}

// PR11391.
struct PR11391 { _Complex float f; } pr11391;
EVAL_EXPR(42, __builtin_constant_p(pr11391.f = 1))

// PR12043
float varfloat;
const float constfloat = 0;
EVAL_EXPR(43, varfloat && constfloat) // expected-error {{must have a constant size}}

// <rdar://problem/10962435>
EVAL_EXPR(45, ((char*)-1) + 1 == 0 ? 1 : -1)
EVAL_EXPR(46, ((char*)-1) + 1 < (char*) -1 ? 1 : -1)
EVAL_EXPR(47, &x < &x + 1 ? 1 : -1)
EVAL_EXPR(48, &x != &x - 1 ? 1 : -1)
EVAL_EXPR(49, &x < &x - 100 ? 1 : -1) // expected-error {{must have a constant size}}

extern struct Test50S Test50;
EVAL_EXPR(50, &Test50 < (struct Test50S*)((unsigned long)&Test50 + 10)) // expected-error {{must have a constant size}}

// <rdar://problem/11874571>
EVAL_EXPR(51, 0 != (float)1e99)

// PR21945
void PR21945() { int i = (({}), 0l); }

void PR24622();
struct PR24622 {} pr24622;
EVAL_EXPR(52, &pr24622 == (void *)&PR24622); // expected-error {{must have a constant size}}

// We evaluate these by providing 2s' complement semantics in constant
// expressions, like we do for integers.
void *PR28739a = (__int128)(unsigned long)-1 + &PR28739a;
void *PR28739b = &PR28739b + (__int128)(unsigned long)-1;
__int128 PR28739c = (&PR28739c + (__int128)(unsigned long)-1) - &PR28739c;
void *PR28739d = &(&PR28739d)[(__int128)(unsigned long)-1];
