// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core,debug.ExprInspection,unix.Malloc,unix.cstring,alpha.unix.cstring,unix.API,osx.API,osx.cocoa.RetainCount -Wno-null-dereference -Wno-tautological-compare -analyzer-store=region -fblocks -verify %s
#define NULL 0
void clang_analyzer_eval(int);
void myFunc();
void myWeakFunc() __attribute__((weak_import));

void testWeakFuncIsNull()
{
  clang_analyzer_eval(myFunc == NULL);  // expected-warning{{FALSE}}
  clang_analyzer_eval(myWeakFunc == NULL);  // expected-warning{{UNKNOWN}}
  if (myWeakFunc == NULL) {
    clang_analyzer_eval(myWeakFunc == NULL);  // expected-warning{{TRUE}}
  } else {
    clang_analyzer_eval(myWeakFunc == NULL);  // expected-warning{{FALSE}}
  }
}

void testWeakFuncIsNot()
{
  clang_analyzer_eval(myWeakFunc == NULL);  // expected-warning{{UNKNOWN}}
  if (!myWeakFunc) {
    clang_analyzer_eval(myWeakFunc == NULL);  // expected-warning{{TRUE}}
  } else {
    clang_analyzer_eval(myWeakFunc == NULL);  // expected-warning{{FALSE}}
  }
}

void testWeakFuncIsTrue()
{
    clang_analyzer_eval(myWeakFunc == NULL);  // expected-warning{{UNKNOWN}}
    if (myWeakFunc) {
        clang_analyzer_eval(myWeakFunc == NULL);  // expected-warning{{FALSE}}
    } else {
        clang_analyzer_eval(myWeakFunc == NULL);  // expected-warning{{TRUE}}
    }
}

//===----------------------------------------------------------------------===
// func.c
//===----------------------------------------------------------------------===
void f(void) __attribute__((weak_import));
void g(void (*fp)(void)) __attribute__((weak_import));

void f(void) {
  void (*p)(void);
  p = f;
  p = &f;
  p();
  (*p)();
}

void g(void (*fp)(void));

void f2() {
  g(f);
}

void f3(void (*f)(void), void (*g)(void)) {
  clang_analyzer_eval(!f); // expected-warning{{UNKNOWN}}
  f();
  clang_analyzer_eval(!f); // expected-warning{{FALSE}}

  clang_analyzer_eval(!g); // expected-warning{{UNKNOWN}}
  (*g)();
  clang_analyzer_eval(!g); // expected-warning{{FALSE}}
}

//===----------------------------------------------------------------------===
// free.c
//===----------------------------------------------------------------------===
void free(void *) __attribute__((weak_import));

void t10 () {
  free((void*)&t10); // expected-warning {{Argument to free() is the address of the function 't10', which is not memory allocated by malloc()}}
}

//===----------------------------------------------------------------------===
// string.c : strnlen()
//===----------------------------------------------------------------------===
typedef typeof(sizeof(int)) size_t;
size_t strlen(const char *s) __attribute__((weak_import));

size_t strlen_fn() {
  return strlen((char*)&strlen_fn); // expected-warning{{Argument to string length function is the address of the function 'strlen_fn', which is not a null-terminated string}}
}

//===----------------------------------------------------------------------===
// unix-fns.c : dispatch_once
//===----------------------------------------------------------------------===
typedef void (^dispatch_block_t)(void);
typedef long dispatch_once_t;
void dispatch_once(dispatch_once_t *predicate, dispatch_block_t block) __attribute__((weak_import));

void test_dispatch_once() {
  dispatch_once_t pred = 0;
  do { if (__builtin_expect(*(&pred), ~0l) != ~0l) dispatch_once((&pred), (^() {})); } while (0); // expected-warning{{Call to 'dispatch_once' uses the local variable 'pred' for the predicate value}}
}
void test_dispatch_once_neg() {
  static dispatch_once_t pred = 0;
  do { if (__builtin_expect(*(&pred), ~0l) != ~0l) dispatch_once((&pred), (^() {})); } while (0); // no-warning
}

//===----------------------------------------------------------------------===
// retain-release-path-notes.m
//===----------------------------------------------------------------------===
typedef struct CFType *CFTypeRef;
CFTypeRef CFCreateSomething() __attribute__((weak_import));
CFTypeRef CFGetSomething() __attribute__((weak_import));

CFTypeRef CFCopyRuleViolation () {
  CFTypeRef object = CFGetSomething();
  return object; // expected-warning{{Object with a +0 retain count returned to caller where a +1 (owning) retain count is expected}}
}

CFTypeRef CFGetRuleViolation () {
  CFTypeRef object = CFCreateSomething(); // expected-warning{{Potential leak of an object stored into 'object'}}
  return object; }
