// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -verify -fsyntax-only %s
// RUN: %clang_cc1 -std=c23 -triple x86_64-scei-ps4 -verify=expected,ps4 -fsyntax-only %s

#define M static
struct S { int a; char b; };
int f1(void);

void test1(void) {
  (void)(constexpr int){1};
  (void)&(static int){42};
  (void)(register int){0};
  (void)(static thread_local int){1};
  (void)(constexpr struct S){1, 'a'};
  (void)(static struct S){2, 'b'};
  (void)(register struct S){3, 'c'};
}

void test2(void) {
  (void)(static constexpr int){1};
  (void)(constexpr static int){2};
  (void)(static thread_local int){3};
  (void)(thread_local static int){4};
  (void)(constexpr register int){5};
  (void)(constexpr static thread_local int){6}; // expected-error {{cannot combine with previous '_Thread_local' declaration specifier}}
}

void test3(void) {
  (void)(static static int){1};             // expected-warning {{duplicate 'static' declaration specifier}}
  (void)(constexpr constexpr int){2};       // expected-warning {{duplicate 'constexpr' declaration specifier}}
  (void)(register register int){3};         // expected-warning {{duplicate 'register' declaration specifier}}
  (void)(thread_local thread_local int){4}; // expected-warning {{duplicate '_Thread_local' declaration specifier}} expected-error {{compound literal with 'thread_local' storage duration at block scope must also specify 'static'}}
}

void test4(void) {
  (void)(register static int){1};       // expected-error {{cannot combine with previous 'register' declaration specifier}}
  (void)(register thread_local int){2}; // expected-error {{cannot combine with previous 'register' declaration specifier}}
  (void)(register constexpr int){3};
  (void)(static register int){4};       // expected-error {{cannot combine with previous 'static' declaration specifier}}
  (void)(register _Atomic int){5};
}

void test5(void) {
  (void)&(thread_local int){1}; // expected-error {{compound literal with 'thread_local' storage duration at block scope must also specify 'static'}}
}

int *a1 = &(register int){42}; // expected-error {{file-scope compound literal specifies 'register'}}

void test6(void) {
  (void)(constexpr volatile int){1}; // expected-error {{constexpr compound literal cannot have type 'const volatile int'}}
  (void)(constexpr _Atomic int){1};  // expected-error {{constexpr compound literal cannot have type 'const _Atomic(int)'}}

  int c;
  (void)(constexpr int[c]){0}; // expected-error {{constexpr compound literal cannot have type 'const int[c]'}}
}

void test7(void) {
  (void)(constexpr int){f1()};          // expected-error {{initializer of compound literal must be a constant expression}}
  (void)(static int){f1()};             // expected-error {{initializer element is not a compile-time constant}}
  (void)(register constexpr int){f1()}; // expected-error {{initializer of compound literal must be a constant expression}}
  (void)(constexpr int){1 / 0};
  // expected-error@-1 {{initializer of compound literal must be a constant expression}}
  // expected-note@-2 {{division by zero}}
  // expected-warning@-3 {{division by zero is undefined}}
}

const int a2 = 1;
const double a3 = 1.0;

void test8(void) {
  (void)(constexpr int){a2};    // expected-error {{initializer of compound literal must be a constant expression}}
  (void)(constexpr double){a3}; // expected-error {{initializer of compound literal must be a constant expression}}

  const int a = 2;
  (void)(constexpr int){a}; // expected-error {{initializer of compound literal must be a constant expression}}

  struct S1 {
    int a;
  };

  (void)(constexpr int){(constexpr int){1}};
  (void)(constexpr struct S1){.a = (constexpr int){1}};
  (void)(constexpr int){(int){1}};
  (void)(constexpr int){(static int){1}};
}

void test9(void) {
  int *a = &(register int){1};   // expected-error {{address of register compound literal requested}}
  int *b = (register int[1]){1}; // expected-error {{address of register compound literal requested}}
  struct S2 { int a; };
  int *c = &(register struct S2){1}.a;                   // expected-error {{address of register compound literal requested}}
  double *d = &__real__ (register _Complex double){1};   // expected-error {{address of register compound literal requested}}
  double *e = &__imag__ (register _Complex double){1};   // expected-error {{address of register compound literal requested}}
  int *f = &_Generic(0, int: (register int){1});         // expected-error {{address of register compound literal requested}}
  int *g = _Generic(0, int: (register int[1]){1});       // expected-error {{address of register compound literal requested}}
  int *h = &_Generic(0, int: (register struct S2){1}).a; // expected-error {{address of register compound literal requested}}
  int *i = &__extension__ (register int){1};             // expected-error {{address of register compound literal requested}}
}

int a5[1];
struct S3 {
  int *a;
};
struct S4 {
  int a[1];
};

int *test10(void) {
  return &((register struct S3){a5}.a[0]);
}

int *test11(void) {
  return &(register struct S4){{1}}.a[0]; // expected-error {{address of register compound literal requested}}
}

enum { E1 = (constexpr int){42} };
enum { E2 = (constexpr int){} };
static int a6[(constexpr int){10}];
static int a7[(constexpr int){} + 1];
static const int *a8 = &(constexpr int){5};
static int a9 = (constexpr int){7};
static_assert(!(constexpr int){});

struct S5 { int a; };
enum { E3 = (constexpr struct S5){42}.a };
enum { E4 = (int)(constexpr double){1.0} };

union U1 {
  int a;
  long b;
};
enum { E5 = (constexpr union U1){.a = 9}.a };
enum {
  E6 = (constexpr union U1){.a = 9}.b
  // expected-error@-1 {{expression is not an integer constant expression}}
  // expected-note@-2 {{read of member 'b' of union with active member 'a' is not allowed in a constant expression}}
};

void test12(void) {
  static int a = (constexpr int){8};
  int b[(constexpr int){3}];
  switch ((constexpr int){1}) {
  case (constexpr int){1}:
    break;
  }
}

void test13(void) {
  (constexpr int){1} = 2;             // expected-error {{read-only variable is not assignable}}
  (constexpr int[1]){1}[0] = 2;       // expected-error {{read-only variable is not assignable}}
  (constexpr struct S){1, 'a'}.a = 2; // expected-error {{read-only variable is not assignable}}
}

int *a10 = &(static int){100};
int *a11 = &(thread_local int){200}; // expected-error {{initializer element is not a compile-time constant}}
const int *a12 = &(constexpr int){300};

void test14(void) {
  const int *a = (constexpr int[]){1, 2, 3};
  int *b = (static int[]){4, 5, 6};
}

int a13 = (thread_local int){1};              // expected-error {{initializer element is not a compile-time constant}}
int a14 = (thread_local static int){1};       // expected-error {{initializer element is not a compile-time constant}}
int a15 = 1 + (thread_local int){1};          // expected-error {{initializer element is not a compile-time constant}}
int a16 = +(thread_local int){1};             // expected-error {{initializer element is not a compile-time constant}}
int a17 = (thread_local struct S5){1}.a;      // expected-error {{initializer element is not a compile-time constant}}
int a18 = *(thread_local int[1]){1};          // expected-error {{initializer element is not a compile-time constant}}

struct S7 {
  unsigned char a;
};

void test15(void) {
  static int a;
  (void)(constexpr int *){&a};     // expected-error {{constexpr pointer initializer is not null}}
  (void)(constexpr struct S7){-1}; // expected-error {{constexpr initializer evaluates to -1 which is not exactly representable in type 'unsigned char'}}
}

int a19[3];
static int *a20 = a19 + (constexpr int){1};
static int *a21 = (constexpr int){0}; // expected-warning {{expression which evaluates to zero treated as a null pointer constant of type 'int *'}}
static int *a22 = (constexpr int *){0};

struct S8 {
  int *restrict a;
};
void test16(void) {
  (void)(constexpr struct S8){0}; // expected-error {{constexpr compound literal cannot have type 'int *restrict'}}
}

void test17(void) {
  (void)(constexpr char[]){"\xFF"};
  (void)(constexpr unsigned char[]){"\xFF"}; // expected-error {{constexpr initializer evaluates to -1 which is not exactly representable in type 'const unsigned char'}}
}

void test18(void) {
  static const int *a = &(constexpr int){1}; // expected-error {{initializer element is not a compile-time constant}}
  static const int *b = &(static constexpr int){1};
}

void f2(int a[sizeof((static int){1})]);
void f3(int a[sizeof((register int){1})]);
void f4(int a[sizeof((constexpr int){1})]);
void f5(int a[sizeof((thread_local static int){1})]);
int f6(void);

void f7(int a[(constexpr int){f6()}]); // expected-error {{initializer of compound literal must be a constant expression}}
void f8(int a[(static int){f6()}]);    // expected-error {{initializer element is not a compile-time constant}}

typedef void F1(int a[(int){f6()}]);
typedef void F2(int a[(thread_local int){1}]); // expected-error {{compound literal with 'thread_local' storage duration at block scope must also specify 'static'}}
void (*a23)(int a[(register int){f6()}]);

typedef int T __attribute__((address_space(1)));
void f9(int a[sizeof((T){0})]);  // expected-error {{compound literal in function scope may not be qualified with an address space}}
int f10(int a[sizeof((T){0})]) { // expected-error {{compound literal in function scope may not be qualified with an address space}}
  return a[0];
}

void test19(void) {
  (static T){0}; // expected-error {{compound literal in function scope may not be qualified with an address space}}
}

int f11(
    typeof(sizeof((register T){0})) a(void)) { // expected-error {{compound literal in function scope may not be qualified with an address space}}
  return a();
}

inline int f12(void) {    // expected-note {{use 'static' to give inline function 'f12' internal linkage}}
  return (static int){1}; // expected-warning {{non-constant static local variable in inline function may be different in different files}}
}

inline int f13(int a[(static int){1}]) { // expected-warning {{non-constant static local variable in inline function may be different in different files}} \
                                         // expected-note {{use 'static' to give inline function 'f13' internal linkage}}
  return a[0];
}

inline int f14(int a[(static int){1}]);

inline int f15(int a[(static const int){1}]) {
  return a[0];
}

extern inline int f16(int a[(static int){1}]) {
  return a[0];
}

static inline int f17(int a[(static int){1}]) {
  return a[0];
}

inline int f18(int b(int a[sizeof((static int){1})])) {
  return 0;
}

inline int f19(int (*a(void))[sizeof((static int){1})]) { // expected-warning {{non-constant static local variable in inline function may be different in different files}}
                                                          // expected-note@-1 {{use 'static' to give inline function 'f19' internal linkage}}
  return 0;
}

inline typeof((static int){1}) f20(void) {
  return 0;
}
typeof((register int){1}) f21(void); // expected-error {{file-scope compound literal specifies 'register'}}

inline int f22(int a) __attribute__((enable_if((static int){1}, "enabled"))) { // expected-warning {{non-constant static local variable in inline function may be different in different files}} \
                                                                               // expected-note {{use 'static' to give inline function 'f22' internal linkage}}
  return a;
}

int f23(int a) __attribute__((enable_if(sizeof((register T){0}), "enabled"))) { // expected-error {{compound literal in function scope may not be qualified with an address space}}
  return a;
}

int f24(int a) __attribute__((enable_if( sizeof((register T){0}), "enabled"))); // expected-error {{compound literal in function scope may not be qualified with an address space}}
int f24(int a) {
  return a;
}

int a24;
int a25 = sizeof((thread_local static int (*)[a24]){0}); // expected-error {{variably modified type declaration not allowed at file scope}}

int *f25(void) {
  return &(static int){1};
}

int *f26(void) {
  return &(thread_local static int){1};
}

int *f27(void) {
  return &(int){1}; // expected-warning {{address of stack memory associated with compound literal '{1}' returned}}
}

const int *f28(void) {
  return &(constexpr int){1}; // expected-warning {{address of stack memory associated with compound literal '{1}' returned}}
}

inline int f29(void) {                 // expected-note {{use 'static' to give inline function 'f29' internal linkage}}
  return (thread_local static int){1}; // expected-warning {{non-constant static local variable in inline function may be different in different files}}
}

void test20(void) {
  (void)(M const int){1};
  (void)(constexpr unsigned long){2};
  (void)sizeof (static int){1};
  (void)sizeof (static thread_local int){1};
  (void)sizeof (constexpr int){1};
}

void test21(void) {
  (void)(const static int){1};   // expected-error {{type name does not allow storage class to be specified}}
  (void)(const int register){1}; // expected-error {{type name does not allow storage class to be specified}}
}

void test22(void) {
  (void)(auto int){1};        // expected-error {{expected expression}}
  (void)(extern int){2};      // expected-error {{expected expression}}
  (void)(typedef int){3};     // expected-error {{expected expression}}
  (void)(__auto_type int){4}; // expected-error {{expected expression}}
  (void)(__thread int){5};    // expected-error {{expected expression}}
}

void test23(void) {
  (void)(static int constexpr){1};     // expected-error {{type name does not allow constexpr specifier to be specified}}
  (void)(static int thread_local){1};  // expected-error {{type name does not allow storage class to be specified}}
  (void)(static int _Thread_local){1}; // expected-error {{type name does not allow storage class to be specified}}
  (void)(static int __thread){1};      // expected-error {{type name does not allow storage class to be specified}}
}

void test24(void) {
  (void)(static int)1;              // expected-error {{type name does not allow storage class to be specified}}
  (void)sizeof(register int);       // expected-error {{type name does not allow storage class to be specified}}
  (void)_Alignof(thread_local int); // expected-error {{type name does not allow storage class to be specified}}
  (void)(constexpr int)1;           // expected-error {{type name does not allow constexpr specifier to be specified}}
}

int a26 = (static)3; // expected-error {{type name requires a specifier or qualifier}}

typedef int A __attribute__((aligned(64)));
A *f30(void) {
  return &(thread_local static A){1}; // ps4-error {{alignment (64) of thread-local compound literal is greater than the maximum supported alignment (32) for thread-local storage on this target}}
}
