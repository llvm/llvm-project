// RUN: %clang_cc1 %s -verify -fsyntax-only -ffreestanding

#include <stddef.h>
#include <stdint.h>

typedef void (* fp)(void);
void foo(void);

// PR clang/3377
fp a[(short int)1] = { foo };

int myArray[5] = {1, 2, 3, 4, 5};
int *myPointer2 = myArray;
int *myPointer = &(myArray[2]);


extern int x;
void *g = &x;
int *h = &x;

struct union_crash
{
    union
    {
    };
};

int test() {
  int a[10];
  int b[10] = a; // expected-error {{array initializer must be an initializer list}}
  int +; // expected-error {{expected identifier or '('}}

  struct union_crash u = { .d = 1 }; // expected-error {{field designator 'd' does not refer to any field in type 'struct union_crash'}}
}


// PR2050
struct cdiff_cmd {
          const char *name;
          unsigned short argc;
          int (*handler)();
};
int cdiff_cmd_open();
struct cdiff_cmd commands[] = {
        {"OPEN", 1, &cdiff_cmd_open }
};

// PR2348
static struct { int z; } s[2];
int *t = &(*s).z;

// PR2349
short *a2(void)
{
  short int b;
  static short *bp = &b; // expected-error {{initializer element is not a compile-time constant}}

  return bp;
}

int pbool(void) {
  typedef const _Bool cbool;
  _Bool pbool1 = (void *) 0;
  cbool pbool2 = &pbool;
  return pbool2;
}


// rdar://5870981
union { float f; unsigned u; } u = { 1.0f };

// rdar://6156694
int f3(int x) { return x; }
typedef void (*vfunc)(void);
void *bar = (vfunc) f3;

// PR2747
struct sym_reg {
        char nc_gpreg;
};
int sym_fw1a_scr[] = {
           ((int)(&((struct sym_reg *)0)->nc_gpreg)) & 0,
           8 * ((int)(&((struct sym_reg *)0)->nc_gpreg))
};

// PR3001
struct s1 s2 = { // expected-error {{variable has incomplete type 'struct s1'}}  \
                 // expected-note {{forward declaration of 'struct s1'}}
    .a = sizeof(struct s3), // expected-error {{invalid application of 'sizeof'}} \
                            // expected-note{{forward declaration of 'struct s3'}}
    .b = bogus // expected-error {{use of undeclared identifier 'bogus'}}
}

// PR3382
char t[] = ("Hello");

// <rdar://problem/6094855>
typedef struct { } empty;

typedef struct {
  empty e;
  int i2;
} st;

st st1 = { .i2 = 1 };

// <rdar://problem/6096826>
struct {
  int a;
  int z[2];
} y = { .z = {} };

int bbb[10];

struct foo2 {
   uintptr_t a;
};

struct foo2 bar2[] = {
   { (intptr_t)bbb }
};

struct foo2 bar3 = { 1, 2 }; // expected-warning{{excess elements in struct initializer}}

int* ptest1 = __builtin_choose_expr(1, (int*)0, (int*)0);

typedef int32_t ivector4 __attribute((vector_size(16)));
ivector4 vtest1 = 1 ? (ivector4){1} : (ivector4){1};
ivector4 vtest2 = __builtin_choose_expr(1, (ivector4){1}, (ivector4){1});

uintptr_t ptrasintadd1 = (uintptr_t)&a - 4;
uintptr_t ptrasintadd2 = (uintptr_t)&a + 4;
uintptr_t ptrasintadd3 = 4 + (uintptr_t)&a;

// PR4285
const wchar_t widestr[] = L"asdf";

// PR5447
const double pr5447 = (0.05 < -1.0) ? -1.0 : 0.0499878;

// PR4386

// None of these are constant initializers, but we implement GCC's old
// behaviour of accepting bar and zed but not foo. GCC's behaviour was
// changed in 2007 (rev 122551), so we should be able to change too one
// day.
int PR4386_bar();
int PR4386_foo() __attribute((weak));
int PR4386_zed();

int PR4386_a = ((void *) PR4386_bar) != 0;
int PR4386_b = ((void *) PR4386_foo) != 0; // expected-error{{initializer element is not a compile-time constant}}
int PR4386_c = ((void *) PR4386_zed) != 0;
int PR4386_zed() __attribute((weak));

// <rdar://problem/10185490> (derived from SPEC vortex benchmark)
typedef char strty[10];
struct vortexstruct { strty s; };
struct vortexstruct vortexvar = { "asdf" };
