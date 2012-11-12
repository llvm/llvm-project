// RUN: %clang_cc1 -fsyntax-only -Wused-but-marked-unused -Wunused-function -Wunneeded-internal-declaration -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wunused %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wall %s

void foo() {}
static void f2() {} 
static void f1() {f2();} // expected-warning{{unused}}

static int f0() { return 17; } // expected-warning{{not needed and will not be emitted}}
int x = sizeof(f0());

static void f3();
extern void f3() { } // expected-warning{{unused}}

// FIXME: This will trigger a warning when it should not.
// Update once PR6281 is fixed.
//inline static void f4();
//void f4() { }

static void __attribute__((used)) f5() {}
static void f6();
static void __attribute__((used)) f6();
static void f6() {};

static void f7(void);
void f8(void(*a0)(void));
void f9(void) { f8(f7); }
static void f7(void) {}

__attribute__((unused)) static void bar(void);
void bar(void) { }

__attribute__((constructor)) static void bar2(void);
void bar2(void) { }

__attribute__((destructor)) static void bar3(void);
void bar3(void) { }

static void f10(void); // expected-warning{{unused}}
static void f10(void);

static void f11(void);
static void f11(void) { }  // expected-warning{{unused}}

static void f12(void) { }  // expected-warning{{unused}}
static void f12(void);

// PR7923
static void unused(void) { unused(); }  // expected-warning{{not needed and will not be emitted}}

// rdar://8728293
static void cleanupMalloc(char * const * const allocation) { }
void f13(void) {
  char * const __attribute__((cleanup(cleanupMalloc))) a;
  (void)a;
}

// rdar://12233989
extern void a(void) __attribute__((unused));
extern void b(void) __attribute__((unused));

void b(void)
{
}
void a(void)
{
  b();
}
