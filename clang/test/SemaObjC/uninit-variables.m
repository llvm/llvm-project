// RUN: %clang_cc1 -Wuninitialized -fsyntax-only -fblocks %s -verify

#include <stdarg.h>

@interface NSObject {} @end
@class NSString;

@interface NSException
+ (void)raise:(NSString *)name format:(NSString *)format, ...;
+ (void)raise:(NSString *)name format:(NSString *)format arguments:(va_list)argList;
- (void)raise;
@end

// Duplicated from uninit-variables.c.
// Test just to ensure the analysis is working.
int test1(void) {
  int x; // expected-note{{initialize the variable 'x' to silence this warning}}
  return x; // expected-warning{{variable 'x' is uninitialized when used here}}
}

// Test ObjC fast enumeration.
void test2(void) {
  id collection = 0;
  for (id obj in collection) {
    if (0 == obj) // no-warning
      break;
  }
}

void test3(void) {
  id collection = 0;
  id obj;
  for (obj in collection) { // no-warning
    if (0 == obj) // no-warning
      break;
  }
}

int test_abort_on_exceptions(int y, NSException *e, NSString *s, int *z, ...) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
  if (y == 1) {
    va_list alist;
    va_start(alist, z);
    [NSException raise:@"Blah" format:@"Blah %@" arguments:alist];
    return x;
  }
  else if (y == 2) {
	[NSException raise:@"Blah" format:s];
	return x;  
  }
  else if (y == 3) {
	[e raise];
	return x;
  }
  return x; // expected-warning {{variable 'x' is uninitialized when used here}}
}
