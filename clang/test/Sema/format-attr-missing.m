// RUN: %clang_cc1 -fsyntax-only -verify -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -Wmissing-format-attribute -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

#include <stdarg.h>

@interface PrintCallee
-(void)printf:(const char *)fmt, ... __attribute__((format(printf, 1, 2)));
-(void)vprintf:(const char *)fmt list:(va_list)ap __attribute__((format(printf, 1, 0)));
@end

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"__attribute__((format(printf, 2, 3))) "
void f1(PrintCallee *p, const char *fmt, int x) // #f1
{
  [p printf:fmt, x]; // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 2, 3)' attribute to the declaration of 'f1'}}
                     // expected-note@#f1 {{'f1' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"__attribute__((format(printf, 2, 3))) "
void f2(PrintCallee *p, const char *fmt, ...) // #f2
{
  va_list ap;
  [p vprintf:fmt list:ap]; // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 2, 3)' attribute to the declaration of 'f2'}}
                           // expected-note@#f2 {{'f2' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"__attribute__((format(printf, 2, 0))) "
void f3(PrintCallee *p, const char *fmt, va_list ap) // #f3
{
  [p vprintf:fmt list:ap]; // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 2, 0)' attribute to the declaration of 'f3'}}
                           // expected-note@#f3 {{'f3' declared here}}
}

__attribute__((format(printf, 1, 2)))
int printf(const char *, ...);
__attribute__((format(printf, 1, 0)))
int vprintf(const char *, va_list ap);

__attribute__((objc_root_class))
@interface PrintCaller
// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:33-[[@LINE+1]]:33}:" __attribute__((format(printf, 1, 2)))"
-(void)f4:(const char *)fmt, ...; // #f4

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:45-[[@LINE+1]]:45}:" __attribute__((format(printf, 1, 0)))"
-(void)f5:(const char *)fmt list:(va_list)ap; // #f5

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:37-[[@LINE+1]]:37}:" __attribute__((format(printf, 1, 2)))"
-(void)f6:(const char *)fmt x:(int)x; // #f6
@end

@implementation PrintCaller
-(void)f4:(const char *)fmt, ... {
  va_list ap;
  va_start(ap, fmt);
  vprintf(fmt, ap); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 2)' attribute to the declaration of 'f4:'}}
                    // expected-note@#f4 {{'f4:' declared here}}
  va_end(ap);
}

-(void)f5:(const char *)fmt list:(va_list)ap {
  vprintf(fmt, ap); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 0)' attribute to the declaration of 'f5:list:'}}
                    // expected-note@#f5 {{'f5:list:' declared here}}
}

-(void)f6:(const char *)fmt x:(int)x {
  printf(fmt, x); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 2)' attribute to the declaration of 'f6:x:'}}
                  // expected-note@#f6 {{'f6:x:' declared here}}
}
@end
