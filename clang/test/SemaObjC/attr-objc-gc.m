// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify %s
static id __attribute((objc_gc(weak))) a;
static id __attribute((objc_gc(strong))) b;

static id __attribute((objc_gc())) c; // expected-error{{'objc_gc' attribute requires parameter 1 to be a string}}
static id __attribute((objc_gc(123))) d; // expected-error{{'objc_gc' attribute requires parameter 1 to be a string}}
static id __attribute((objc_gc(foo, 456))) e; // expected-error{{attribute takes one argument}}
static id __attribute((objc_gc(hello))) f; // expected-warning{{'objc_gc' attribute argument not supported: 'hello'}}

static int __attribute__((objc_gc(weak))) g; // expected-warning {{'objc_gc' only applies to pointer types; type here is 'int'}}

static __weak int h; // expected-warning {{'__weak' only applies to pointer types; type here is 'int'}}

// TODO: it would be great if this reported as __weak
#define WEAK __weak
static WEAK int h; // expected-warning {{'objc_gc' only applies to pointer types; type here is 'int'}}

/* expected-warning {{'__weak' only applies to pointer types; type here is 'int'}}*/ static __we\
ak int i;

// rdar://problem/9126213
void test2(id __attribute((objc_gc(strong))) *strong,
           id __attribute((objc_gc(weak))) *weak) {
  void *opaque;
  opaque = strong;
  strong = opaque;

  opaque = weak;
  weak = opaque;
}
