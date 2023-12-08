// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify %s
static id __attribute((objc_gc(weak))) a;
static id __attribute((objc_gc(strong))) b;

static id __attribute((objc_gc())) c; // expected-error{{'objc_gc' attribute requires a string}}
static id __attribute((objc_gc(123))) d; // expected-error{{'objc_gc' attribute requires a string}}
static id __attribute((objc_gc(foo, 456))) e; // expected-error{{'objc_gc' attribute takes one argument}}
static id __attribute((objc_gc(hello))) f; // expected-warning{{'objc_gc' attribute argument not supported: 'hello'}}

static int __attribute__((objc_gc(weak))) g; // expected-warning {{'objc_gc' only applies to pointer types; type here is 'int'}}

static __weak int h; // expected-warning {{'__weak' only applies to Objective-C object or block pointer types; type here is 'int'}}

// TODO: it would be great if this reported as __weak
#define WEAK __weak
static WEAK int h; // expected-warning {{'objc_ownership' only applies to Objective-C object or block pointer types; type here is 'int'}}

/* expected-warning {{'__weak' only applies to Objective-C object or block pointer types; type here is 'int'}}*/ static __we\
ak int i;

static id [[clang::objc_gc(weak)]] j;
[[clang::objc_gc(weak)]] static id k; // expected-warning {{applying attribute 'objc_gc' to a declaration is deprecated; apply it to the type instead}}
static id l [[clang::objc_gc(weak)]]; // expected-warning {{applying attribute 'objc_gc' to a declaration is deprecated; apply it to the type instead}}

void test2(id __attribute((objc_gc(strong))) *strong,
           id __attribute((objc_gc(weak))) *weak) {
  void *opaque;
  opaque = strong;
  strong = opaque;

  opaque = weak;
  weak = opaque;
}
