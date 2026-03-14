// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fdiagnostics-parseable-fixits -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK-NOT: fix-it:

@class NSString;
extern void NSLog(NSString *format, ...);
int printf(const char * restrict, ...) ;


void test_object_correction (id x) {  
  printf("%d", x); // expected-warning{{format specifies type 'int' but the argument has type 'id'}}
  printf("%s", x); // expected-warning{{format specifies type 'char *' but the argument has type 'id'}}
  printf("%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'id'}}
}


// Old-style Core Foundation types do not have __attribute__((NSObject)).
// This is okay, but we won't suggest a fixit; arbitrary structure pointers may
// not be objects.
typedef const struct __CFString * CFStringRef;

void test_cf_object_correction (CFStringRef x) {
  NSLog(@"%@", x); // no-warning

  NSLog(@"%d", x); // expected-warning{{format specifies type 'int' but the argument has type 'CFStringRef'}}
  NSLog(@"%s", x); // expected-warning{{format specifies type 'char *' but the argument has type 'CFStringRef'}}
  NSLog(@"%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'CFStringRef'}}
}

