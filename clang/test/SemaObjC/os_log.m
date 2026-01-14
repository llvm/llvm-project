// RUN: %clang_cc1 -verify %s

struct S {
  int a[4];
};

struct S s;
_Complex float cf;

void test_builtin_os_log_invalid_arg(void *buf) {
  __builtin_os_log_format(buf, "%*.*f", s, 5, 1.3); // expected-error {{field width should have type 'int', but argument has type 'struct S'}}
  __builtin_os_log_format(buf, "%*.*f", 1, s, 1.3); // expected-error {{field precision should have type 'int', but argument has type 'struct S'}}
  __builtin_os_log_format(buf, "%*.*f", 1, 5, s); // expected-error {{format specifies type 'double' but the argument has type 'struct S'}}

  __builtin_os_log_format(buf, "%*.*f", cf, 5, 1.3); // expected-error {{field width should have type 'int', but argument has type '_Complex float'}}
  __builtin_os_log_format(buf, "%*.*f", 1, cf, 1.3); // expected-error {{field precision should have type 'int', but argument has type '_Complex float'}}
  __builtin_os_log_format(buf, "%*.*f", 1, 5, cf); // expected-error {{format specifies type 'double' but the argument has type '_Complex float'}}

  __builtin_os_log_format(buf, "%*.*f", (void *)0, 5, 1.3); // expected-warning {{field width should have type 'int', but argument has type 'void *'}}
  __builtin_os_log_format(buf, "%*.*f", 1, (void *)0, 1.3); // expected-warning {{field precision should have type 'int', but argument has type 'void *'}}
  __builtin_os_log_format(buf, "%*.*f", 1, 5, (void *)0); // expected-warning {{format specifies type 'double' but the argument has type 'void *'}}
}
