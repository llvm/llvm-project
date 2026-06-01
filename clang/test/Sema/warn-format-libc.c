// RUN: %clang_cc1 -std=c11 -fsyntax-only -Wformat -verify=expected %s
// RUN: %clang_cc1 -std=c11 -fsyntax-only -Wall -verify=expected %s
// RUN: %clang_cc1 -std=c11 -fsyntax-only -Wformat-libc -verify=expected %s
// RUN: %clang_cc1 -std=c11 -fsyntax-only -Wformat -Wno-format-libc -verify=disabled %s
// RUN: %clang_cc1 -std=c11 -fsyntax-only -verify=plain %s
// disabled-no-diagnostics
// plain-no-diagnostics

typedef struct FILE FILE;

int scanf(const char *restrict format, ...);
FILE *fopen(const char *restrict path, const char *restrict mode);

int wrapped_scanf(const char *format, ...)
    __attribute__((format(scanf, 1, 2)));
FILE *wrapped_fopen(const char *path, const char *mode)
    __attribute__((diagnose_as_builtin(fopen, 1, 2)));

void test_scanf(void) {
  char buf[8];
  char one[1];

  scanf("%s", buf); // expected-warning {{'%s' conversion used without a field width, destination buffer has 8 elements}}
  scanf("%[a-z]", buf); // expected-warning {{'%[' conversion used without a field width, destination buffer has 8 elements}}
  wrapped_scanf("%s", buf); // expected-warning {{'%s' conversion used without a field width, destination buffer has 8 elements}}
  scanf("%s", one); // expected-warning {{'%s' conversion used without a field width, destination buffer has 1 element}}

  scanf("%7s", buf);
  scanf("%7[a-z]", buf);
}

void test_scanf_unknown(char *ptr) {
  scanf("%s", ptr);
}

void test_fopen(const char *path, const char *mode) {
  fopen(path, "r");
  fopen(path, "w+b");
  fopen(path, "rb+cmxe");
  fopen(path, "w+, ccs=UTF-8");
  fopen(path, mode);

  fopen(path, "oo"); // expected-warning {{invalid fopen mode string 'oo'}}
  fopen(path, "rx"); // expected-warning {{invalid fopen mode string 'rx'}}
  fopen(path, "r++"); // expected-warning {{invalid fopen mode string 'r++'}}
  wrapped_fopen(path, ""); // expected-warning {{invalid fopen mode string ''}}
}
