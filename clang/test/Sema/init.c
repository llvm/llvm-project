// RUN: %clang_cc1 %s -Wno-pointer-to-int-cast -verify -fsyntax-only -ffreestanding

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

int test(void) {
  int a[10];
  int b[10] = a; // expected-error {{array initializer must be an initializer list}}
  int +; // expected-error {{expected identifier or '('}}

  struct union_crash u = { .d = 1 }; // expected-error {{field designator 'd' does not refer to any field in type 'struct union_crash'}}
}


// PR2050
struct cdiff_cmd {
          const char *name;
          unsigned short argc;
          int (*handler)(void);
};
int cdiff_cmd_open(void);
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


union { float f; unsigned u; } u = { 1.0f };

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

typedef struct { } empty;

typedef struct {
  empty e;
  int i2;
} st;

st st1 = { .i2 = 1 };

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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexcess-initializers"
struct foo2 bar3_silent = {1, 2};
#pragma clang diagnostic pop

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
int PR4386_bar(void);
int PR4386_foo(void) __attribute((weak));
int PR4386_zed(void);

int PR4386_a = ((void *) PR4386_bar) != 0;
int PR4386_b = ((void *) PR4386_foo) != 0; // expected-error{{initializer element is not a compile-time constant}}
int PR4386_c = ((void *) PR4386_zed) != 0;
int PR4386_zed(void) __attribute((weak));

// (derived from SPEC vortex benchmark)
typedef char strty[10];
struct vortexstruct { strty s; };
struct vortexstruct vortexvar = { "asdf" };

typedef struct { uintptr_t x : 2; } StructWithBitfield;
StructWithBitfield bitfieldvar = { (uintptr_t)&bitfieldvar }; // expected-error {{initializer element is not a compile-time constant}}

// PR45157
struct PR4517_foo {
  int x;
};
struct PR4517_bar {
  struct PR4517_foo foo;
};
const struct PR4517_foo my_foo = {.x = 42};
struct PR4517_bar my_bar = {
    .foo = my_foo, // no-warning
};
struct PR4517_bar my_bar2 = (struct PR4517_bar){
    .foo = my_foo, // no-warning
};
struct PR4517_bar my_bar3 = {
    my_foo, // no-warning
};
struct PR4517_bar my_bar4 = (struct PR4517_bar){
    my_foo // no-warning
};
extern const struct PR4517_foo my_foo2;
struct PR4517_bar my_bar5 = {
  .foo = my_foo2, // expected-error {{initializer element is not a compile-time constant}}
};
const struct PR4517_foo my_foo3 = {.x = my_foo.x};
int PR4517_a[2] = {0, 1};
const int PR4517_ca[2] = {0, 1};
int PR4517_idx = 0;
const int PR4517_idxc = 1;
int PR4517_x1 = PR4517_a[PR4517_idx]; // expected-error {{initializer element is not a compile-time constant}}
int PR4517_x2 = PR4517_a[PR4517_idxc]; // expected-error {{initializer element is not a compile-time constant}}
int PR4517_x3 = PR4517_a[0]; // expected-error {{initializer element is not a compile-time constant}}
int PR4517_y1 = PR4517_ca[PR4517_idx]; // expected-error {{initializer element is not a compile-time constant}}
int PR4517_y2 = PR4517_ca[PR4517_idxc]; // no-warning
int PR4517_y3 = PR4517_ca[0]; // no-warning
union PR4517_u {
    int x;
    float y;
};
const union PR4517_u u1 = {4.0f};
const union PR4517_u u2 = u1; // no-warning
const union PR4517_u u3 = {u1.y}; // expected-error {{initializer element is not a compile-time constant}}
