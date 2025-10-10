// RUN: %clang_cc1 -triple arm64-apple-macosx15 -fblocks -ffeature-availability=feature1:1 -ffeature-availability=feature2:0 -ffeature-availability=feature3:on -fsyntax-only -Wunreachable-code -verify=expected,redecl %s
// RUN: %clang_cc1 -triple arm64-apple-macosx15 -fblocks -ffeature-availability=feature1:1 -ffeature-availability=feature2:0 -ffeature-availability=feature3:on -fsyntax-only -Wunreachable-code -Wno-domain-availability-redeclaration -verify=expected %s
// RUN: %clang_cc1 -triple arm64-apple-macosx15 -fblocks -fsyntax-only -Wunreachable-code -verify=expected,redecl -DUSE_DOMAIN %s
// RUN: not %clang_cc1 -triple arm64-apple-macosx15 -fblocks -fsyntax-only -Wunreachable-code -DUSE_DOMAIN -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

#include <availability_domain.h>

#define AVAIL 0
#define UNAVAIL 1
#define INVALID 2

#define FEATURE_AVAILABLE(domain_name) \
  __attribute__((availability(domain : domain_name, AVAIL)))

#define FEATURE_UNAVAILABLE(domain_name) \
  __attribute__((availability(domain : domain_name, UNAVAIL)))

#define MY_FEATURE_AVAILABLE(domain_name) \
  /* abc */ __attribute__((availability(domain /*abc*/ : domain_name, 0))) // comment.

#define VISIBILITY
#define MY_FEATURE_AVAILABLE2(domain_name) \
  VISIBILITY __attribute__((availability(domain : domain_name, 0)))

#define MY_FEATURE_AVAILABLE3(domain_name) FEATURE_AVAILABLE(domain_name)

#define AVAIL_ARGS domain: deprecated_feature1, 0
#define MY_FEATURE_AVAILABLE4(domain_name) \
  __attribute__((availability(AVAIL_ARGS)))

#define MY_FEATURE_AVAILABLE5(domain_name, function) \
  FEATURE_AVAILABLE(domain_name) \
  function

#define DEPRECATED_FEATURE1_UNAVAILABLE __attribute__((availability(domain: deprecated_feature1, 1)))

#ifdef USE_DOMAIN
int pred1(void);
CLANG_ENABLED_AVAILABILITY_DOMAIN(feature1);
CLANG_DISABLED_AVAILABILITY_DOMAIN(feature2);
CLANG_ENABLED_AVAILABILITY_DOMAIN(feature3);
CLANG_DYNAMIC_AVAILABILITY_DOMAIN(feature4, pred1);
CLANG_ALWAYS_ENABLED_AVAILABILITY_DOMAIN(feature5);
__attribute__((deprecated))
CLANG_ALWAYS_ENABLED_AVAILABILITY_DOMAIN(deprecated_feature1);
__attribute__((deprecated))
CLANG_ENABLED_AVAILABILITY_DOMAIN(deprecated_feature2);
#endif

#pragma clang attribute push (__attribute__((availability(domain:feature1, AVAIL))), apply_to=any(function))
void func12(void);
#pragma clang attribute pop

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
__attribute__((availability(domain:feature5, AVAIL))) void func21(void);
__attribute__((availability(domain:feature5, UNAVAIL))) void func22(void);

__attribute__((visibility("hidden")))
__attribute__((availability(domain:deprecated_feature1, AVAIL)))
void deprecated_feature1_avail_func1(void);
// expected-warning@-2 {{availability domain 'deprecated_feature1' is deprecated}}
// expected-warning@-3 {{attribute has no effect because 'deprecated_feature1' is always available}}
// CHECK: fix-it:"{{.*}}feature-availability.c":{[[@LINE-4]]:1-[[@LINE-4]]:65}:""

__attribute__((availability(domain:deprecated_feature1, AVAIL), visibility("hidden")))
void deprecated_feature1_avail_func2(void);
// expected-warning@-2 {{availability domain 'deprecated_feature1' is deprecated}}
// expected-warning@-3 {{attribute has no effect because 'deprecated_feature1' is always available}}
// CHECK: fix-it:"{{.*}}feature-availability.c":{[[@LINE-4]]:16-[[@LINE-4]]:64}:""

__attribute__((visibility("hidden"), availability(domain:deprecated_feature1, AVAIL)))
void deprecated_feature1_avail_func3(void);
// expected-warning@-2 {{availability domain 'deprecated_feature1' is deprecated}}
// expected-warning@-3 {{attribute has no effect because 'deprecated_feature1' is always available}}
// CHECK: fix-it:"{{.*}}feature-availability.c":{[[@LINE-4]]:36-[[@LINE-4]]:85}:""

FEATURE_AVAILABLE(deprecated_feature1)
void deprecated_feature1_avail_func4(void);
// expected-warning@-2 {{availability domain 'deprecated_feature1' is deprecated}}
// expected-warning@-3 {{attribute has no effect because 'deprecated_feature1' is always available}}
// CHECK: fix-it:"{{.*}}feature-availability.c":{[[@LINE-4]]:1-[[@LINE-4]]:39}:""

MY_FEATURE_AVAILABLE(deprecated_feature1)
void deprecated_feature1_avail_func5(void);
// expected-warning@-2 {{availability domain 'deprecated_feature1' is deprecated}}
// expected-warning@-3 {{attribute has no effect because 'deprecated_feature1' is always available}}
// CHECK: fix-it:"{{.*}}feature-availability.c":{[[@LINE-4]]:1-[[@LINE-4]]:42}:""

MY_FEATURE_AVAILABLE2(deprecated_feature1)
void deprecated_feature1_avail_func6(void);
// expected-warning@-2 {{availability domain 'deprecated_feature1' is deprecated}}
// expected-warning@-3 {{attribute has no effect because 'deprecated_feature1' is always available}}
// CHECK-NOT: fix-it:

MY_FEATURE_AVAILABLE3(deprecated_feature1)
void deprecated_feature1_avail_func7(void);
// expected-warning@-2 {{availability domain 'deprecated_feature1' is deprecated}}
// expected-warning@-3 {{attribute has no effect because 'deprecated_feature1' is always available}}
// CHECK-NOT: fix-it:

MY_FEATURE_AVAILABLE4(deprecated_feature1)
void deprecated_feature1_avail_func8(void);
// expected-warning@-2 {{availability domain 'deprecated_feature1' is deprecated}}
// expected-warning@-3 {{attribute has no effect because 'deprecated_feature1' is always available}}
// CHECK-NOT: fix-it:

MY_FEATURE_AVAILABLE5(deprecated_feature1, void deprecated_feature1_avail_func9(void);)
// expected-warning@-1 {{availability domain 'deprecated_feature1' is deprecated}}
// expected-warning@-2 {{attribute has no effect because 'deprecated_feature1' is always available}}
// CHECK-NOT: fix-it:

FEATURE_AVAILABLE(deprecated_feature1)
void deprecated_feature1_avail_func10(void);
// expected-warning@-2 {{availability domain 'deprecated_feature1' is deprecated}}
// expected-warning@-3 {{attribute has no effect because 'deprecated_feature1' is always available}}
// CHECK: fix-it:"{{.*}}feature-availability.c":{[[@LINE-4]]:1-[[@LINE-4]]:39}:""

__attribute__((visibility("hidden")))
__attribute__((availability(domain:deprecated_feature1, UNAVAIL)))
void deprecated_feature1_unavail_func1(void);
// expected-warning@-2 {{availability domain 'deprecated_feature1' is deprecated}}
// expected-warning@-3 {{declaration is permanently unavailable because 'deprecated_feature1' is always available}}
// CHECK: fix-it:"{{.*}}feature-availability.c":{[[@LINE-4]]:16-[[@LINE-4]]:65}:"unavailable"

__attribute__((availability(domain:deprecated_feature1, UNAVAIL), visibility("hidden")))
void deprecated_feature1_unavail_func2(void);
// expected-warning@-2 {{availability domain 'deprecated_feature1' is deprecated}}
// expected-warning@-3 {{declaration is permanently unavailable because 'deprecated_feature1' is always available}}
// CHECK: fix-it:"{{.*}}feature-availability.c":{[[@LINE-4]]:16-[[@LINE-4]]:65}:"unavailable"

__attribute__((visibility("hidden"), availability(domain:deprecated_feature1, UNAVAIL)))
void deprecated_feature1_unavail_func3(void);
// expected-warning@-2 {{availability domain 'deprecated_feature1' is deprecated}}
// expected-warning@-3 {{declaration is permanently unavailable because 'deprecated_feature1' is always available}}
// CHECK: fix-it:"{{.*}}feature-availability.c":{[[@LINE-4]]:38-[[@LINE-4]]:87}:"unavailable"

FEATURE_UNAVAILABLE(deprecated_feature1)
void deprecated_feature1_unavail_func4(void);
// expected-warning@-2 {{availability domain 'deprecated_feature1' is deprecated}}
// expected-warning@-3 {{declaration is permanently unavailable because 'deprecated_feature1' is always available}}
// CHECK: fix-it:"{{.*}}feature-availability.c":{[[@LINE-4]]:1-[[@LINE-4]]:41}:"__attribute__((unavailable))"

DEPRECATED_FEATURE1_UNAVAILABLE
void deprecated_feature1_unavail_func5(void);
// expected-warning@-2 {{availability domain 'deprecated_feature1' is deprecated}}
// expected-warning@-3 {{declaration is permanently unavailable because 'deprecated_feature1' is always available}}
// CHECK: fix-it:"{{.*}}feature-availability.c":{[[@LINE-4]]:1-[[@LINE-4]]:32}:"__attribute__((unavailable))"

__attribute__((availability(domain:deprecated_feature2, AVAIL)))
void deprecated_feature2_unavail_func1(void);
// expected-warning@-2 {{availability domain 'deprecated_feature2' is deprecated}}

#endif

void test_unreachable_code(void) {
  if (__builtin_available(domain:feature1)) {
  } else {
    // Warning -Wunreachable-code isn't emitted.
    (void)2;
  }
}

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

int *g12 = &g0; // expected-error {{cannot use 'g0' because feature 'feature1' is unavailable in this context}}
int g7 = sizeof(g0); // expected-error {{cannot use 'g0' because feature 'feature1' is unavailable in this context}}
__attribute__((availability(domain:feature1, AVAIL))) int g6 = sizeof(g0);
__attribute__((availability(domain:feature1, UNAVAIL))) int g8 = sizeof(g0); // expected-error {{cannot use 'g0' because feature 'feature1' is unavailable in this context}}
__attribute__((availability(domain:feature2, AVAIL))) int g9 = sizeof(g0); // expected-error {{cannot use 'g0' because feature 'feature1' is unavailable in this context}}
void (*fp0)(void) = func6; // expected-error {{cannot use 'func6' because feature 'feature1' is unavailable in this context}}
void (* __attribute__((availability(domain:feature1, AVAIL))) fp1)(void) = func6;
void (* __attribute__((availability(domain:feature1, UNAVAIL))) fp2)(void) = func6; // expected-error {{cannot use 'func6' because feature 'feature1' is unavailable in this context}}
void (* __attribute__((availability(domain:feature2, AVAIL))) fp3)(void) = func6; // expected-error {{cannot use 'func6' because feature 'feature1' is unavailable in this context}}

void func6(void);
__attribute__((availability(domain:feature1, AVAIL))) void func6(void); // expected-note {{is incompatible with __attribute__((availability(domain:feature1, 0)))}}
__attribute__((availability(domain:feature1, UNAVAIL))) void func6(void); // expected-error {{cannot merge incompatible feature attribute to this decl}} expected-note {{feature attribute __attribute__((availability(domain:feature1, 1)))}}
__attribute__((availability(domain:feature1, AVAIL))) void func8(void); // redecl-error {{new domain availability attributes cannot be added to redeclarations}}

int g0;
__attribute__((availability(domain:feature1, AVAIL))) int g0; // expected-note {{is incompatible with __attribute__((availability(domain:feature1, 0)))}}
__attribute__((availability(domain:feature1, UNAVAIL))) int g0; // expected-error {{cannot merge incompatible feature attribute to this decl}} expected-note {{feature attribute __attribute__((availability(domain:feature1, 1)))}}
__attribute__((availability(domain:feature1, AVAIL))) int g2;// redecl-error {{new domain availability attributes cannot be added to redeclarations}}

typedef int INT0 __attribute__((availability(domain:feature2, AVAIL)));
typedef INT0 INT1 __attribute__((availability(domain:feature2, AVAIL)));
typedef INT0 INT2 __attribute__((availability(domain:feature2, UNAVAIL))); // expected-error {{cannot use 'INT0' because feature 'feature2' is unavailable in this context}}
typedef INT0 INT3 __attribute__((availability(domain:feature1, AVAIL))); // expected-error {{cannot use 'INT0' because feature 'feature2' is unavailable in this context}}

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
  struct S0 s0; // expected-error {{cannot use 'S0' because feature 'feature1' is unavailable in this context}}
  int i0 __attribute__((availability(domain:feature1, AVAIL))); // expected-error {{feature attributes cannot be applied to struct members}}
};

struct S0 g10; // expected-error {{cannot use 'S0' because feature 'feature1' is unavailable in this context}}
__attribute__((availability(domain:feature1, AVAIL))) struct S0 g11;

void test0(void) {
  func12(); // expected-error {{cannot use 'func12' because feature 'feature1' is unavailable in this context}}
  func7(); // expected-error {{cannot use 'func7' because feature 'feature1' is available in this context}}}
  func19(); // expected-error {{cannot use 'func19' because feature 'feature1' is unavailable in this context}}

  if (__builtin_available(domain:feature1, domain:feature2)) // expected-error {{cannot pass a domain argument along with other arguments}}
    ;

  if (__builtin_available(domain:feature1)) {
    func12();
    func7(); // expected-error {{cannot use 'func7' because feature 'feature1' is available in this context}}}
    func13(); // expected-error {{cannot use 'func13' because feature 'feature2' is unavailable in this context}}
    if (__builtin_available(domain:feature2)) {
      func13();
      func9(); // expected-error {{cannot use 'func9' because feature 'feature2' is available in this context}}}
      func12();
    } else {
      func13(); // expected-error {{cannot use 'func13' because feature 'feature2' is unavailable in this context}}
      func9();
      func12();
    }
  } else {
    func12(); // expected-error {{cannot use 'func12' because feature 'feature1' is unavailable in this context}}
    func7();
  }
}

 __attribute__((availability(domain:feature1, AVAIL)))
void test1(void) {
  func12();
  func7(); // expected-error {{cannot use 'func7' because feature 'feature1' is available in this context}}}
}

 __attribute__((availability(domain:feature1, UNAVAIL)))
void test2(void) {
  func12(); // expected-error {{cannot use 'func12' because feature 'feature1' is unavailable in this context}}
  func7();
}

__attribute__((availability(domain:feature3, AVAIL)))
void test3(void) {
  ^ {
    func12(); // expected-error {{cannot use 'func12' because feature 'feature1' is unavailable in this context}}
    func7(); // expected-error {{cannot use 'func7' because feature 'feature1' is available in this context}}}
    func20();
  }();

  if (__builtin_available(domain:feature1)) {
    ^{
      func12();
      func7(); // expected-error {{cannot use 'func7' because feature 'feature1' is available in this context}}}
      func20();
      if (__builtin_available(domain:feature2)) {
        func13();
        func9(); // expected-error {{cannot use 'func9' because feature 'feature2' is available in this context}}}
      } else {
        func13(); // expected-error {{cannot use 'func13' because feature 'feature2' is unavailable in this context}}
        func9();
      }
    }();
  } else {
    ^{
      func12(); // expected-error {{cannot use 'func12' because feature 'feature1' is unavailable in this context}}
      func7();
      func20();
    }();
  }
}

void test4(struct S0 *s0) { // expected-error {{cannot use 'S0' because feature 'feature1' is unavailable in this context}}
  g11.i0 = 0; // expected-error {{cannot use 'g11' because feature 'feature1' is unavailable in this context}} expected-error {{cannot use 'i0' because feature 'feature1' is unavailable in this context}}
}

void test5(int c) {
  if (c > 100)
    goto label0; // expected-error {{cannot jump from this goto statement to its label}}
  else if (c > 50)
    goto label1; // expected-error {{cannot jump from this goto statement to its label}}
  if (__builtin_available(domain:feature1)) { // expected-note 2 {{jump enters controlled statement of if available}}
    label0:
    ;
  } else {
    if (c > 80)
      goto label2;
    label1:
    ;
    label2:
    ;
  }
}

void test6(void) {
  if (__builtin_available(domain:feature1)) {
    if (__builtin_available(domain:feature2)) {
      func15();
      func16(); // expected-error {{cannot use 'func16' because feature 'feature2' is available in this context}}}
      func17(); // expected-error {{cannot use 'func17' because feature 'feature1' is available in this context}}}
      func18(); // expected-error {{cannot use 'func18' because feature 'feature1' is available in this context}}} expected-error {{cannot use 'func18' because feature 'feature2' is available in this context}}}
    } else {
      func15(); // expected-error {{cannot use 'func15' because feature 'feature2' is unavailable in this context}}
      func16();
      func17(); // expected-error {{cannot use 'func17' because feature 'feature1' is available in this context}}} expected-error {{cannot use 'func17' because feature 'feature2' is unavailable in this context}}
      func18(); // expected-error {{cannot use 'func18' because feature 'feature1' is available in this context}}}
    }
  } else {
    if (__builtin_available(domain:feature2)) {
      func15(); // expected-error {{cannot use 'func15' because feature 'feature1' is unavailable in this context}}
      func16(); // expected-error {{cannot use 'func16' because feature 'feature1' is unavailable in this context}} expected-error {{cannot use 'func16' because feature 'feature2' is available in this context}}}
      func17();
      func18(); // expected-error {{cannot use 'func18' because feature 'feature2' is available in this context}}}
    } else {
      func15(); // expected-error {{cannot use 'func15' because feature 'feature1' is unavailable in this context}} expected-error {{cannot use 'func15' because feature 'feature2' is unavailable in this context}}
      func16(); // expected-error {{cannot use 'func16' because feature 'feature1' is unavailable in this context}}
      func17(); // expected-error {{cannot use 'func17' because feature 'feature2' is unavailable in this context}}
      func18();
    }
  }
}

void test7(void) {
  enum E e; // expected-error {{cannot use 'E' because feature 'feature1' is unavailable in this context}}
  struct S0 s0; // expected-error {{cannot use 'S0' because feature 'feature1' is unavailable in this context}}

  if (__builtin_available(domain:feature1)) {
    enum E e;
    e = EA;
    e = EB; // expected-error {{cannot use 'EB' because feature 'feature2' is unavailable in this context}}

    switch (e) {
    case EA: {
      if (__builtin_available(domain:feature2))
        e = EB;
      break;
    }
    case EB: // no diagnostic
      e = EB; // expected-error {{cannot use 'EB' because feature 'feature2' is unavailable in this context}}
      break;
    }
  }

  if (__builtin_available(domain:feature2)) {
    enum E e; // expected-error {{cannot use 'E' because feature 'feature1' is unavailable in this context}}
    e = EA; // expected-error {{cannot use 'EA' because feature 'feature1' is unavailable in this context}}
    e = EB; // expected-error {{cannot use 'EB' because feature 'feature1' is unavailable in this context}}
  }
}

#ifdef USE_DOMAIN
void test8(void) {
  func21();
  func22(); // expected-error {{cannot use 'func22' because feature 'feature5' is available in this context}}

  if (__builtin_available(domain:feature5)) {
    func21();
    func22(); // expected-error {{cannot use 'func22' because feature 'feature5' is available in this context}}
  } else {
    func21();
    func22();
  }
}

void test_deprecated_feature(void) {
  if (__builtin_available(domain:deprecated_feature1))
    // expected-warning@-1 {{availability domain 'deprecated_feature1' is deprecated}}
    // expected-warning@-2 {{unnecessary check for 'deprecated_feature1'; this expression will always evaluate to true}}
    ;

  if (!__builtin_available(domain:deprecated_feature1))
    // expected-warning@-1 {{availability domain 'deprecated_feature1' is deprecated}}
    // expected-warning@-2 {{unnecessary check for 'deprecated_feature1'; this expression will always evaluate to true}}
    ;

  if (__builtin_available(domain:deprecated_feature2))
    // expected-warning@-1 {{availability domain 'deprecated_feature2' is deprecated}}
    ;
}
#endif
