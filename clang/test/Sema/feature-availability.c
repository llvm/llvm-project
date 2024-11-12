// RUN: %clang_cc1 -triple arm64-apple-macosx15 -fblocks -ffeature-availability=feature1:1 -ffeature-availability=feature2:0 -ffeature-availability=feature3:on -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple arm64-apple-macosx15 -fblocks -fsyntax-only -verify -DUSE_DOMAIN %s

#include <feature-availability.h>

#define AVAIL 0
#define UNAVAIL 1
#define INVALID 2

#ifdef USE_DOMAIN
int pred1(void);
static struct __AvailabilityDomain __feature1 __attribute__((availability_domain(feature1))) = {__AVAILABILITY_DOMAIN_ENABLED, 0};
static struct __AvailabilityDomain __feature2 __attribute__((availability_domain(feature2))) = {__AVAILABILITY_DOMAIN_DISABLED, 0};
static struct __AvailabilityDomain __feature3 __attribute__((availability_domain(feature3))) = {__AVAILABILITY_DOMAIN_ENABLED, 0};
static struct __AvailabilityDomain __feature4 __attribute__((availability_domain(feature4))) = {__AVAILABILITY_DOMAIN_DYNAMIC, pred1};
#endif

__attribute__((availability(domain:feature1, AVAIL))) void func12(void);
__attribute__((availability(domain:feature2, AVAIL))) void func13(void);
__attribute__((availability(domain:feature1, AVAIL))) void func6(void);
__attribute__((availability(domain:feature1, UNAVAIL))) void func7(void);
__attribute__((availability(domain:feature2, AVAIL))) void func8(void);
__attribute__((availability(domain:feature2, UNAVAIL))) void func9(void);
__attribute__((availability(domain:feature3, AVAIL))) void func20(void);
__attribute__((availability(domain:feature1, AVAIL))) int g0;
__attribute__((availability(domain:feature1, UNAVAIL))) int g1;
__attribute__((availability(domain:feature2, AVAIL))) int g2;
__attribute__((availability(domain:feature2, UNAVAIL))) int g3;
#ifdef USE_DOMAIN
__attribute__((availability(domain:feature4, AVAIL))) void func10(void);
__attribute__((availability(domain:feature4, UNAVAIL))) void func11(void);
__attribute__((availability(domain:feature4, AVAIL))) int g4;
__attribute__((availability(domain:feature4, UNAVAIL))) int g5;
#endif

__attribute__((availability(domain:2, AVAIL))) void func5(void); // expected-error {{expected a domain name}}
__attribute__((availability(domain, AVAIL))) void func4(void); // expected-error {{expected ':'}}
__attribute__((availability(domain:, AVAIL))) void func0(void); // expected-error {{expected a domain name}}
__attribute__((availability(domain:nonexistent, AVAIL))) void func1(void); // expected-error {{cannot find definition of feature attribute 'nonexistent'}}
__attribute__((availability(domain:feature1, INVALID))) void func2(void); // expected-error {{invalid argument 2: must evaluate to 0 or 1}}
__attribute__((availability(domain:feature1, "1"))) void func3(void); // expected-error {{expected <numeric_constant>}}

__attribute__((availability(domain:feature1, AVAIL), availability(domain:feature1, UNAVAIL))) void func14(void); // expected-error {{cannot add feature availability to this decl}} expected-note {{feature attribute __attribute__((availability(domain:feature1, 1)))}} expected-note {{is incompatible with __attribute__((availability(domain:feature1, 0)))}}
__attribute__((availability(domain:feature1, AVAIL), availability(domain:feature2, AVAIL))) void func15(void);
__attribute__((availability(domain:feature1, AVAIL), availability(domain:feature2, UNAVAIL))) void func16(void);
__attribute__((availability(domain:feature1, UNAVAIL), availability(domain:feature2, AVAIL))) void func17(void);
__attribute__((availability(domain:feature1, UNAVAIL), availability(domain:feature2, UNAVAIL))) void func18(void);

__attribute__((availability(macosx,introduced=10), availability(domain:feature1, AVAIL))) void func19(void);

int *g12 = &g0; // expected-error {{use of 'g0' requires feature 'feature1' to be available}}
int g7 = sizeof(g0); // expected-error {{use of 'g0' requires feature 'feature1' to be available}}
__attribute__((availability(domain:feature1, AVAIL))) int g6 = sizeof(g0);
__attribute__((availability(domain:feature1, UNAVAIL))) int g8 = sizeof(g0); // expected-error {{use of 'g0' requires feature 'feature1' to be available}}
__attribute__((availability(domain:feature2, AVAIL))) int g9 = sizeof(g0); // expected-error {{use of 'g0' requires feature 'feature1' to be available}}
void (*fp0)(void) = func6; // expected-error {{use of 'func6' requires feature 'feature1' to be available}}
void (* __attribute__((availability(domain:feature1, AVAIL))) fp1)(void) = func6;
void (* __attribute__((availability(domain:feature1, UNAVAIL))) fp2)(void) = func6; // expected-error {{use of 'func6' requires feature 'feature1' to be available}}
void (* __attribute__((availability(domain:feature2, AVAIL))) fp3)(void) = func6; // expected-error {{use of 'func6' requires feature 'feature1' to be available}}

void func6(void);
__attribute__((availability(domain:feature1, AVAIL))) void func6(void); // expected-note {{is incompatible with __attribute__((availability(domain:feature1, 0)))}}
__attribute__((availability(domain:feature1, UNAVAIL))) void func6(void); // expected-error {{cannot merge incompatible feature attribute to this decl}} expected-note {{feature attribute __attribute__((availability(domain:feature1, 1)))}}
__attribute__((availability(domain:feature1, AVAIL))) void func8(void); // expected-error {{new feature attributes cannot be added to redeclarations}}

int g0;
__attribute__((availability(domain:feature1, AVAIL))) int g0; // expected-note {{is incompatible with __attribute__((availability(domain:feature1, 0)))}}
__attribute__((availability(domain:feature1, UNAVAIL))) int g0; // expected-error {{cannot merge incompatible feature attribute to this decl}} expected-note {{feature attribute __attribute__((availability(domain:feature1, 1)))}}
__attribute__((availability(domain:feature1, AVAIL))) int g2;// expected-error {{new feature attributes cannot be added to redeclarations}}

typedef int INT0 __attribute__((availability(domain:feature2, AVAIL)));
typedef INT0 INT1 __attribute__((availability(domain:feature2, AVAIL)));
typedef INT0 INT2 __attribute__((availability(domain:feature2, UNAVAIL))); // expected-error {{use of 'INT0' requires feature 'feature2' to be available}}
typedef INT0 INT3 __attribute__((availability(domain:feature1, AVAIL))); // expected-error {{use of 'INT0' requires feature 'feature2' to be available}}

enum __attribute__((availability(domain:feature1, AVAIL))) E {
  EA,
  EB __attribute__((availability(domain:feature2, AVAIL))),
};

struct __attribute__((availability(domain:feature1, AVAIL))) S0 {
  int i0;
};

struct __attribute__((availability(domain:feature1, AVAIL))) S1 {
  struct S0 s0;
};

struct S2 {
  struct S0 s0; // expected-error {{use of 'S0' requires feature 'feature1' to be available}}
  int i0 __attribute__((availability(domain:feature1, AVAIL))); // expected-error {{feature attributes cannot be applied to struct members}}
};

struct S0 g10; // expected-error {{use of 'S0' requires feature 'feature1' to be available}}
__attribute__((availability(domain:feature1, AVAIL))) struct S0 g11;

void test0(void) {
  func12(); // expected-error {{use of 'func12' requires feature 'feature1' to be available}}
  func7(); // expected-error {{use of 'func7' requires feature 'feature1' to be unavailable}}
  func19(); // expected-error {{use of 'func19' requires feature 'feature1' to be available}}

  if (__builtin_available(domain:feature1, domain:feature2)) // expected-error {{cannot pass a domain argument along with other arguments}}
    ;

  if (__builtin_available(domain:feature1)) {
    func12();
    func7(); // expected-error {{use of 'func7' requires feature 'feature1' to be unavailable}}
    func13(); // expected-error {{use of 'func13' requires feature 'feature2' to be available}}
    if (__builtin_available(domain:feature2)) {
      func13();
      func9(); // expected-error {{use of 'func9' requires feature 'feature2' to be unavailable}}
      func12();
    } else {
      func13(); // expected-error {{use of 'func13' requires feature 'feature2' to be available}}
      func9();
      func12();
    }
  } else {
    func12(); // expected-error {{use of 'func12' requires feature 'feature1' to be available}}
    func7();
  }
}

 __attribute__((availability(domain:feature1, AVAIL)))
void test1(void) {
  func12();
  func7(); // expected-error {{use of 'func7' requires feature 'feature1' to be unavailable}}
}

 __attribute__((availability(domain:feature1, UNAVAIL)))
void test2(void) {
  func12(); // expected-error {{use of 'func12' requires feature 'feature1' to be available}}
  func7();
}

__attribute__((availability(domain:feature3, AVAIL)))
void test3(void) {
  ^ {
    func12(); // expected-error {{use of 'func12' requires feature 'feature1' to be available}}
    func7(); // expected-error {{use of 'func7' requires feature 'feature1' to be unavailable}}
    func20();
  }();

  if (__builtin_available(domain:feature1)) {
    ^{
      func12();
      func7(); // expected-error {{use of 'func7' requires feature 'feature1' to be unavailable}}
      func20();
      if (__builtin_available(domain:feature2)) {
        func13();
        func9(); // expected-error {{use of 'func9' requires feature 'feature2' to be unavailable}}
      } else {
        func13(); // expected-error {{use of 'func13' requires feature 'feature2' to be available}}
        func9();
      }
    }();
  } else {
    ^{
      func12(); // expected-error {{use of 'func12' requires feature 'feature1' to be available}}
      func7();
      func20();
    }();
  }
}

void test4(struct S0 *s0) { // expected-error {{use of 'S0' requires feature 'feature1' to be available}}
  g11.i0 = 0; // expected-error {{use of 'g11' requires feature 'feature1' to be available}} expected-error {{use of 'i0' requires feature 'feature1' to be available}}
}

void test5(void) {
  if (__builtin_available(domain:feature1))
    label0: // expected-error {{labels cannot appear in regions conditionally guarded by features}}
      ;
  label1:
    ;
}

void test6(void) {
  if (__builtin_available(domain:feature1)) {
    if (__builtin_available(domain:feature2)) {
      func15();
      func16(); // expected-error {{use of 'func16' requires feature 'feature2' to be unavailable}}
      func17(); // expected-error {{use of 'func17' requires feature 'feature1' to be unavailable}}
      func18(); // expected-error {{use of 'func18' requires feature 'feature1' to be unavailable}} expected-error {{use of 'func18' requires feature 'feature2' to be unavailable}}
    } else {
      func15(); // expected-error {{use of 'func15' requires feature 'feature2' to be available}}
      func16();
      func17(); // expected-error {{use of 'func17' requires feature 'feature1' to be unavailable}} expected-error {{use of 'func17' requires feature 'feature2' to be available}}
      func18(); // expected-error {{use of 'func18' requires feature 'feature1' to be unavailable}}
    }
  } else {
    if (__builtin_available(domain:feature2)) {
      func15(); // expected-error {{use of 'func15' requires feature 'feature1' to be available}}
      func16(); // expected-error {{use of 'func16' requires feature 'feature1' to be available}} expected-error {{use of 'func16' requires feature 'feature2' to be unavailable}}
      func17();
      func18(); // expected-error {{use of 'func18' requires feature 'feature2' to be unavailable}}
    } else {
      func15(); // expected-error {{use of 'func15' requires feature 'feature1' to be available}} expected-error {{use of 'func15' requires feature 'feature2' to be available}}
      func16(); // expected-error {{use of 'func16' requires feature 'feature1' to be available}}
      func17(); // expected-error {{use of 'func17' requires feature 'feature2' to be available}}
      func18();
    }
  }
}

void test7(void) {
  enum E e; // expected-error {{use of 'E' requires feature 'feature1' to be available}}
  struct S0 s0; // expected-error {{use of 'S0' requires feature 'feature1' to be available}}

  if (__builtin_available(domain:feature1)) {
    enum E e;
    e = EA;
    e = EB; // expected-error {{use of 'EB' requires feature 'feature2' to be available}}
  }

  if (__builtin_available(domain:feature2)) {
    enum E e; // expected-error {{use of 'E' requires feature 'feature1' to be available}}
    e = EA; // expected-error {{use of 'EA' requires feature 'feature1' to be available}}
    e = EB; // expected-error {{use of 'EB' requires feature 'feature1' to be available}}
  }
}
