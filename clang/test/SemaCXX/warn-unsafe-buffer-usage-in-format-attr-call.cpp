// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage \
// RUN:            -verify=expected,libc %s
// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage \
// RUN:            -Wno-unsafe-buffer-usage-in-libc-call -verify %s
// RUN: %clang_cc1 -std=c++20 -Wno-all \
// RUN:            -Wunsafe-buffer-usage-in-format-attr-call -verify %s
// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage \
// RUN:            -Wno-unsafe-buffer-usage-in-format-attr-call -verify=libc %s

typedef __SIZE_TYPE__ size_t;

int printf(const char *, ...) __attribute__((__format__(__printf__, 1, 2)));
size_t strlen(const char *);
void log_message(const char *, ...)
    __attribute__((__format__(__printf__, 1, 2)));
void read_input(const char *, ...)
    __attribute__((__format__(__scanf__, 1, 2)));

void f(char *s, int *x) {
  log_message("%s", s); // expected-warning{{formatting function 'log_message' is unsafe}} \
                           expected-note{{string argument is not guaranteed to be null-terminated}}
  read_input("%d", x);  // expected-warning{{formatting function 'read_input' is unsafe}}
  printf("%s", s);      // libc-warning{{function 'printf' is unsafe}} \
                           libc-note{{string argument is not guaranteed to be null-terminated}}
  strlen(s);            // libc-warning{{function 'strlen' is unsafe}}
}

void g(char *s) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-libc-call"
  log_message("%s", s); // expected-warning{{formatting function 'log_message' is unsafe}} \
                           expected-note{{string argument is not guaranteed to be null-terminated}}
  printf("%s", s);
  strlen(s);
#pragma clang diagnostic pop
}
