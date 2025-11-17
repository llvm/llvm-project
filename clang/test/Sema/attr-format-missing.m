// RUN: %clang_cc1 -fsyntax-only -verify -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -Wmissing-format-attribute -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

#include <stdarg.h>

@interface Print
-(void)printf:(const char *)fmt, ... __attribute__((format(printf, 1, 2)));
-(void)vprintf:(const char *)fmt list:(va_list)ap __attribute__((format(printf, 1, 0)));
@end

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"__attribute__((format(printf, 2, 3))) "
void f1(Print *p, const char *fmt, int x) // #f1
{
    [p printf:fmt, x]; // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 2, 3)' attribute to the declaration of 'f1'}}
                       // expected-note@#f1 {{'f1' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"__attribute__((format(printf, 2, 3))) "
void f2(Print *p, const char *fmt, ...) // #f2
{
    va_list ap;
    [p vprintf:fmt list:ap]; // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 2, 3)' attribute to the declaration of 'f2'}}
                             // expected-note@#f2 {{'f2' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"__attribute__((format(printf, 2, 0))) "
void f3(Print *p, const char *fmt, va_list ap) // #f3
{
    [p vprintf:fmt list:ap]; // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 2, 0)' attribute to the declaration of 'f3'}}
                             // expected-note@#f3 {{'f3' declared here}}
}
