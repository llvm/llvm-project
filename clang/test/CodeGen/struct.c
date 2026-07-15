// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o -

struct  {
  int x;
  int y;
} point;

void fn1(void) {
  point.x = 42;
}

/* Nested member */
struct  {
  struct {
    int a;
    int b;
  } p1;
} point2;

void fn2(void) {
  point2.p1.a = 42;
}

/* Indirect reference */
typedef struct __sf {
 unsigned char *c;
 short flags;
} F;

typedef struct __sf2 {
  F *ff;
} F2;

int fn3(F2 *c) {
  if (c->ff->c >= 0)
    return 1;
  else
    return 0;
}

/* Nested structs */
typedef struct NA {
  int data;
  struct NA *next;
} NA;
void f1(void) {  NA a; }

typedef struct NB {
  int d1;
  struct _B2 {
    int d2;
    struct NB *n2;
  } B2;
} NB;

void f2(void) { NB b; }

extern NB *f3(void);
void f4(void) {
  f3()->d1 = 42;
}

void f5(void) {
  (f3())->d1 = 42;
}

/* Function calls */
typedef struct {
  int location;
  int length;
} range;
extern range f6(void);
void f7(void) {
  range r = f6();
}

/* Member expressions */
typedef struct {
  range range1;
  range range2;
} rangepair;

void f8(void) {
  rangepair p;

  range r = p.range1;
}

void f9(range *p) {
  range r = *p;
}

void f10(range *p) {
  range r = p[0];
}

/* _Bool types */

struct _w {
  short a,b;
  short c,d;
  short e,f;
  short g;

  unsigned int h,i;

  _Bool j,k;
} ws;

/* Implicit casts (due to typedefs) */
typedef struct _a {
  int a;
} a;

void f11(void) {
  struct _a a1;
  a a2;
    
  a1 = a2;
  a2 = a1;
}

/* Implicit casts (due to const) */
void f12(void) {
  struct _a a1;
  const struct _a a2;

  a1 = a2;
}

/* struct initialization */
struct a13 {int b; int c;};
struct a13 c13 = {5};
typedef struct a13 a13;
struct a14 { short a; int b; } x = {1, 1};

/* flexible array members */
struct a15 {char a; int b[];} c15;
void a16(void) {c15.a = 1;}

/* compound literals */
void f13(void) {
  a13 x; x = (a13){1,2};
}

/* va_arg */
int f14(int i, ...) {
  __builtin_va_list l;
  __builtin_va_start(l,i);
  a13 b = __builtin_va_arg(l, a13);
  int c = __builtin_va_arg(l, a13).c;
  return b.b;
}

/* Attribute packed */
struct __attribute__((packed)) S2839 { double a[19];  signed char b; } s2839[5];

struct __attribute__((packed)) SS { long double a; char b; } SS;


/* As lvalue */

int f15(void) {
  extern range f15_ext(void);
  return f15_ext().location;
}

range f16(void) {
  extern rangepair f16_ext(void);
  return f16_ext().range1;
}

int f17(void) {
  extern range f17_ext(void);
  range r;
  return (r = f17_ext()).location;
}

range f18(void) {
  extern rangepair f18_ext(void);
  rangepair rp;
  return (rp = f18_ext()).range1;
}



// Complex forward reference of struct.
struct f19S;
extern struct f19T {
  struct f19S (*p)(void);
} t;
struct f19S { int i; };
void f19(void) {
  t.p();
}

