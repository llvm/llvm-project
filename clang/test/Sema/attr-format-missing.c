// RUN: %clang_cc1 -fsyntax-only -verify -std=c23 -Wmissing-format-attribute %s
// RUN: %clang_cc1 -fsyntax-only -std=c23 -Wmissing-format-attribute -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

typedef unsigned long size_t;
typedef long ssize_t;
typedef __builtin_va_list va_list;

[[gnu::format(printf, 1, 2)]]
int printf(const char *, ...);

[[gnu::format(scanf, 1, 2)]]
int scanf(const char *, ...);

[[gnu::format(printf, 1, 0)]]
int vprintf(const char *, va_list);

[[gnu::format(scanf, 1, 0)]]
int vscanf(const char *, va_list);

[[gnu::format(printf, 2, 0)]]
int vsprintf(char *, const char *, va_list);

struct tm { unsigned i; };
[[gnu::format(strftime, 3, 0)]]
size_t strftime(char *, size_t, const char *, const struct tm *);

[[gnu::format(strfmon, 3, 4)]]
ssize_t strfmon(char *, size_t, const char *, ...);

[[gnu::format_matches(printf, 1, "%d %f \"'")]]
int custom_print(const char *, va_list);

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format(printf, 1, 0)]] "
void f1(const char *fmt, va_list args) // #f1
{
  vprintf(fmt, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 0)' attribute to the declaration of 'f1'}}
                      // expected-note@#f1 {{'f1' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format(scanf, 1, 0)]] "
void f2(const char *fmt, va_list args) // #f2
{
  vscanf(fmt, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(scanf, 1, 0)' attribute to the declaration of 'f2'}}
                     // expected-note@#f2 {{'f2' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format(printf, 1, 2)]] "
void f3(const char *fmt, ... /* args */) // #f3
{
  va_list args;
  vprintf(fmt, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 2)' attribute to the declaration of 'f3'}}
                      // expected-note@#f3 {{'f3' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format(scanf, 1, 2)]] "
void f4(const char *fmt, ... /* args */) // #f4
{
  va_list args;
  vscanf(fmt, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(scanf, 1, 2)' attribute to the declaration of 'f4'}}
                     // expected-note@#f4 {{'f4' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+2]]:1-[[@LINE+2]]:1}:"{{\[\[}}gnu::format(printf, 2, 3)]] "
[[gnu::format(printf, 1, 3)]]
void f5(char *out, const char *format, ... /* args */) // #f5
{
  va_list args;
  vsprintf(out, format, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 2, 3)' attribute to the declaration of 'f5'}}
                               // expected-note@#f5 {{'f5' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+2]]:1-[[@LINE+2]]:1}:"{{\[\[}}gnu::format(printf, 2, 3)]] "
[[gnu::format(scanf, 1, 3)]]
void f6(char *out, const char *format, ... /* args */) // #f6
{
  va_list args;
  vsprintf(out, format, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 2, 3)' attribute to the declaration of 'f6'}}
                               // expected-note@#f6 {{'f6' declared here}}
}

// Ok, out is not passed to print functions.
void f7(char* out, ... /* args */)
{
  va_list args;

  const char *ch = "format";
  vprintf(ch, args);
  vprintf("test", args);
}

// Ok, out is not passed to print functions.
void f8(char *out, va_list args)
{
  const char *ch = "format";
  vprintf(ch, args);
  vprintf("test", args);
}

// Ok, out is not passed to scan functions.
void f9(va_list args)
{
  const char *ch = "format";
  vscanf(ch, args);
  vscanf("test", args);
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+2]]:1-[[@LINE+2]]:1}:"{{\[\[}}gnu::format(scanf, 1, 2)]] "
// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format(printf, 1, 2)]] "
void f10(const char *out, ... /* args */) // #f10
{
  va_list args;
  vscanf(out, args);  // expected-warning {{diagnostic behavior may be improved by adding the 'format(scanf, 1, 2)' attribute to the declaration of 'f10'}}
                      // expected-note@#f10 {{'f10' declared here}}
  vprintf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 2)' attribute to the declaration of 'f10'}}
                      // expected-note@#f10 {{'f10' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format(printf, 1, 2)]] "
void f11(const char out[], ... /* args */) // #f11
{
  va_list args;
  char ch[10] = "format";
  vprintf(ch, args);
  vsprintf(ch, out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 2)' attribute to the declaration of 'f11'}}
                            // expected-note@#f11 {{'f11' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format(printf, 1, 0)]] "
void f12(char* out) // #f12
{
  va_list args;
  const char *ch = "format";
  vsprintf(out, ch, args);
  vprintf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 0)' attribute to the declaration of 'f12'}}
                      // expected-note@#f12 {{'f12' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+2]]:1-[[@LINE+2]]:1}:"{{\[\[}}gnu::format(scanf, 1, 2)]] "
// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format(printf, 1, 2)]] "
void f13(char *out, ... /* args */) // #f13
{
  va_list args;
  vscanf(out, args);  // expected-warning {{diagnostic behavior may be improved by adding the 'format(scanf, 1, 2)' attribute to the declaration of 'f13'}}
                      // expected-note@#f13 {{'f13' declared here}}
  vprintf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 2)' attribute to the declaration of 'f13'}}
                      // expected-note@#f13 {{'f13' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+2]]:1-[[@LINE+2]]:1}:"{{\[\[}}gnu::format(scanf, 1, 0)]] "
// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format(printf, 1, 0)]] "
void f14(char *out, va_list args) // #f14
{
  vscanf(out, args);  // expected-warning {{diagnostic behavior may be improved by adding the 'format(scanf, 1, 0)' attribute to the declaration of 'f14'}}
                      // expected-note@#f14 {{'f14' declared here}}
  vprintf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 0)' attribute to the declaration of 'f14'}}
                      // expected-note@#f14 {{'f14' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format(scanf, 1, 2)]] "
void f15(char *out, ... /* args */) // #f15
{
  va_list args;
  vscanf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(scanf, 1, 2)' attribute to the declaration of 'f15'}}
                     // expected-note@#f15 {{'f15' declared here}}
  vscanf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(scanf, 1, 2)' attribute to the declaration of 'f15'}}
                     // expected-note@#f15 {{'f15' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+2]]:1-[[@LINE+2]]:1}:"{{\[\[}}gnu::format(printf, 1, 3)]] "
// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format(printf, 2, 3)]] "
void f16(char *ch, const char *out, ... /* args */) // #f16
{
  va_list args;
  vprintf(ch, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 3)' attribute to the declaration of 'f16'}}
                      // expected-note@#f16 {{'f16' declared here}}
  vprintf(out, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 2, 3)' attribute to the declaration of 'f16'}}
                      // expected-note@#f16 {{'f16' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format(printf, 1, 2)]] "
void f17(const char *a, ...) // #f17
{
	va_list ap;
	const char *const b = a;
	vprintf(b, ap); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 2)' attribute to the declaration of 'f17'}}
                  // expected-note@#f17 {{'f17' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format(printf, 1, 2)]] "
void f18(char *fmt, unsigned x, unsigned y, unsigned z) // #f18
{
  printf(fmt, x, y, z); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 2)' attribute to the declaration of 'f18'}}
                        // expected-note@#f18 {{'f18' declared here}}
}

void f19(char *fmt, unsigned x, unsigned y, unsigned z) // #f19
{
  // Arguments are not passed in the same order.
  printf(fmt, x, z, y);
}

void f20(char *out, ... /* args */)
{
  printf(out, 1); // No warning, arguments are not passed to printf.
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format(strftime, 3, 0)]] "
void f21(char *out, const size_t len, const char *format) // #f21
{
  struct tm tm_arg;
  tm_arg.i = 0;
  strftime(out, len, format, &tm_arg); // expected-warning {{diagnostic behavior may be improved by adding the 'format(strftime, 3, 0)' attribute to the declaration of 'f21'}}
                                       // expected-note@#f21 {{'f21' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format(strfmon, 3, 4)]] "
void f22(char *out, const size_t len, const char *format, int x, int y) // #f22
{
  strfmon(out, len, format, x, y); // expected-warning {{diagnostic behavior may be improved by adding the 'format(strfmon, 3, 4)' attribute to the declaration of 'f22'}}
                                   // expected-note@#f22 {{'f22' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format(printf, 1, 2)]] "
void f23(const char *fmt, ... /* args */); // #f23

void f23(const char *fmt, ... /* args */)
{
  va_list args;
  vprintf(fmt, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format(printf, 1, 2)' attribute to the declaration of 'f23'}}
                      // expected-note@#f23 {{'f23' declared here}}
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:1-[[@LINE+1]]:1}:"{{\[\[}}gnu::format_matches(printf, 1, \"%d %f \\\"'\")]] "
void f24(const char *fmt, ...) // #f24
{
  va_list args;
  custom_print(fmt, args); // expected-warning {{diagnostic behavior may be improved by adding the 'format_matches(printf, 1, "%d %f \"'")' attribute to the declaration of 'f24'}}
                           // expected-note@#f24 {{'f24' declared here}}
}
