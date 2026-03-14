// RUN: %clang_cc1 -triple x86_64-apple-macosx -fsyntax-only -verify -x objective-c++ %s -o /dev/null
// REQUIRES: asserts

struct objc_class {
  unsigned long long bits;
};
typedef struct objc_class *Class;
static void f(Class c) { (void)(c->bits & RW_HAS_OVERFLOW_REFCOUNT); }
// expected-error@-1{{use of undeclared identifier 'RW_HAS_OVERFLOW_REFCOUNT}}
