// REQUIRES: system-darwin
// RUN: rm -f %t.log
// RUN: env RC_DEBUG_OPTIONS=1 \
// RUN:     CC_LOG_DIAGNOSTICS=1 CC_LOG_DIAGNOSTICS_FILE=%t.log \
// RUN: %clang -target x86_64-apple-darwin -fsyntax-only %s
// RUN: FileCheck %s < %t.log

#include <stdarg.h>
int printf(const char *restrict, ...);
int scanf(const char * restrict, ...);
@class NSString, Protocol;
extern void NSLog(NSString *format, ...);

void testInvalidNoPrintable(int *a) {
  // CHECK: <string>invalid conversion specifier &apos;\u25b9&apos;</string>
  // CHECK: <string>invalid conversion specifier &apos;\u25b9&apos;</string>
  // CHECK: <string>invalid conversion specifier &apos;\U00010348&apos;</string>
  // CHECK: <string>invalid conversion specifier &apos;\U00010348&apos;</string>
  // CHECK: <string>invalid conversion specifier &apos;\xe2&apos;</string>
  // CHECK: <string>invalid conversion specifier &apos;\u25b9&apos;</string>
  // CHECK: <string>invalid conversion specifier &apos;\u25b9&apos;</string>
  // CHECK: <string>invalid conversion specifier &apos;\U00010348&apos;</string>
  // CHECK: <string>invalid conversion specifier &apos;\U00010348&apos;</string>
  // CHECK: <string>invalid conversion specifier &apos;\xe2&apos;</string>
  // CHECK: <string>invalid conversion specifier &apos;\u25b9&apos;</string>
  // CHECK: <string>invalid conversion specifier &apos;\u25b9&apos;</string>
  // CHECK: <string>invalid conversion specifier &apos;\U00010348&apos;</string>
  // CHECK: <string>invalid conversion specifier &apos;\U00010348&apos;</string>
  // CHECK: <string>invalid conversion specifier &apos;\xe2&apos;</string>
  printf("%\u25B9");
  printf("%\xE2\x96\xB9");
  printf("%\U00010348");
  printf("%\xF0\x90\x8D\x88");
  printf("%\xe2");
  NSLog(@"%\u25B9");
  NSLog(@"%\xE2\x96\xB9");
  NSLog(@"%\U00010348");
  NSLog(@"%\xF0\x90\x8D\x88");
  NSLog(@"%\xe2");
  scanf("%\u25B9", a);
  scanf("%\xE2\x96\xB9", a);
  scanf("%\U00010348", a);
  scanf("%\xF0\x90\x8D\x88", a);
  scanf("%\xe2", a);
}
