// RUN: %clang_cc1 -fsyntax-only -verify %s

#include <stdarg.h>
#include <stddef.h>
#define __need_wint_t
#include <stddef.h> // For wint_t and wchar_t

int printf(const char *restrict, ...);

@interface NSString
@end

void test_os_log_format(const char *pc, int i, void *p, void *buf) {
  __builtin_os_log_format(buf, "");
  __builtin_os_log_format(buf, "%d"); // expected-warning {{more '%' conversions than data arguments}}
  __builtin_os_log_format(buf, "%d", i);
  __builtin_os_log_format(buf, "%P", p); // expected-warning {{using '%P' format specifier without precision}}
  __builtin_os_log_format(buf, "%.10P", p);
  __builtin_os_log_format(buf, "%.*P", p); // expected-warning {{field precision should have type 'int', but argument has type 'void *'}}
  __builtin_os_log_format(buf, "%.*P", i, p);
  __builtin_os_log_format(buf, "%.*P", i, i); // expected-warning {{format specifies type 'void *' but the argument has type 'int'}}
  __builtin_os_log_format(buf, "%n");         // expected-error {{os_log() '%n' format specifier is not allowed}}
  __builtin_os_log_format(buf, pc);           // expected-error {{os_log() format argument is not a string constant}}

  printf("%{private}s", pc); // expected-warning {{using 'private' format specifier annotation outside of os_log()/os_trace()}}
  __builtin_os_log_format(buf, "%{private}s", pc);

  __builtin_os_log_format_buffer_size("no-args");
  __builtin_os_log_format(buf, "%s", "hi");

  wchar_t wc = 'a';
  __builtin_os_log_format(buf, "%C", wc);
  printf("%C", wc);
  wchar_t wcs[] = {'a', 0};
  __builtin_os_log_format(buf, "%S", wcs);
  printf("%S", wcs);

  struct { char data[0x100]; } toobig;
  __builtin_os_log_format(buf, "%s", toobig); // expected-error {{os_log() argument 2 is too big (256 bytes, max 255)}}

  __builtin_os_log_format(buf, "%{mask.xyz}s", "abc");
  __builtin_os_log_format(buf, "%{mask.}s", "abc"); // expected-error {{mask type size must be between 1-byte and 8-bytes}}
  __builtin_os_log_format(buf, "%{mask.abcdefghi}s", "abc"); // expected-error {{mask type size must be between 1-byte and 8-bytes}}
}

// Test os_log_format primitive with ObjC string literal format argument.
void test_objc(const char *pc, int i, void *p, void *buf, NSString *nss, id obj) {
  __builtin_os_log_format(buf, @"");
  __builtin_os_log_format(buf, @"%d"); // expected-warning {{more '%' conversions than data arguments}}
  __builtin_os_log_format(buf, @"%d", i);

  __builtin_os_log_format(buf, @"%P", p); // expected-warning {{using '%P' format specifier without precision}}
  __builtin_os_log_format(buf, @"%.10P", p);
  __builtin_os_log_format(buf, @"%.*P", p); // expected-warning {{field precision should have type 'int', but argument has type 'void *'}}
  __builtin_os_log_format(buf, @"%.*P", i, p);
  __builtin_os_log_format(buf, @"%.*P", i, i); // expected-warning {{format specifies type 'void *' but the argument has type 'int'}}
  __builtin_os_log_format(buf, @"%.8P", nss);    // expected-warning {{using '%P' format specifier with an Objective-C pointer results in dumping runtime object structure, not object value}}
  __builtin_os_log_format(buf, @"%.*P", i, obj); // expected-warning {{using '%P' format specifier with an Objective-C pointer results in dumping runtime object structure, not object value}}

  __builtin_os_log_format(buf, @"%{private}s", pc);
  __builtin_os_log_format(buf, @"%@", nss);
}

// Test the os_log format attribute.
void MyOSLog(const char *format, ...) __attribute__((format(os_log, 1, 2)));
void test_attribute(void *p) {
  MyOSLog("%s\n", "Hello");
  MyOSLog("%d");    // expected-warning {{more '%' conversions than data arguments}}
  MyOSLog("%P", p); // expected-warning {{using '%P' format specifier without precision}}
}
