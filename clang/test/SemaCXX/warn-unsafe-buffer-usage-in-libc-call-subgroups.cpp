// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage-in-libc-call \
// RUN:            -verify=expected,exam %s
// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage \
// RUN:            -Wno-unsafe-buffer-usage-in-libc-call-string-examination \
// RUN:            -verify=expected %s
// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage-in-libc-call-memory \
// RUN:            -verify=mem %s
// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage \
// RUN:            -Wno-unsafe-buffer-usage-in-libc-call \
// RUN:            -Wunsafe-buffer-usage-in-libc-call-memory -verify=mem %s
// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage \
// RUN:            -Wno-unsafe-buffer-usage-in-libc-call -verify=none %s
// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage-in-libc-call \
// RUN:            %s 2>&1 | FileCheck %s

// none-no-diagnostics

typedef __SIZE_TYPE__ size_t;
struct FILE;

int atoi(const char *);
char *strcpy(char *, const char *);
size_t strlen(const char *);
size_t wcslen(const wchar_t *);
void *memcpy(void *, const void *, size_t);
void *memset(void *, int, size_t);
void bzero(void *, size_t);
char *fgets(char *, int, FILE *);
int sprintf(char *, const char *, ...);
int sscanf(const char *, const char *, ...);
void qsort(void *, size_t, size_t, int (*)(const void *, const void *));

void calls(char *p, const char *q, const wchar_t *w, size_t n, FILE *f,
           int *x) {
  atoi(q);          // expected-warning{{function 'atoi' is unsafe}}
  strcpy(p, q);     // expected-warning{{function 'strcpy' is unsafe}}
  strlen(q);        // exam-warning{{function 'strlen' is unsafe}}
  wcslen(w);        // exam-warning{{function 'wcslen' is unsafe}}
  memcpy(p, q, n);  // expected-warning{{function 'memcpy' is unsafe}} mem-warning{{function 'memcpy' is unsafe}}
  memset(p, 0, n);  // expected-warning{{function 'memset' is unsafe}} mem-warning{{function 'memset' is unsafe}}
  bzero(p, n);      // expected-warning{{function 'bzero' is unsafe}} mem-warning{{function 'bzero' is unsafe}}
  fgets(p, 4, f);   // expected-warning{{function 'fgets' is unsafe}}
  sprintf(p, "%d", 1); // expected-warning{{function 'sprintf' is unsafe}} \
                          expected-note{{change to 'snprintf' for explicit bounds checking}}
  sscanf(q, "%d", x);  // expected-warning{{function 'sscanf' is unsafe}}
  qsort(p, n, 1, nullptr); // expected-warning{{function 'qsort' is unsafe}}
}

void pragma(char *p, const char *q, size_t n) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-libc-call-string-examination"
  strlen(q);
  memcpy(p, q, n); // expected-warning{{function 'memcpy' is unsafe}} mem-warning{{function 'memcpy' is unsafe}}
#pragma clang diagnostic pop
  strlen(q); // exam-warning{{function 'strlen' is unsafe}}
}

// CHECK-DAG: function 'atoi' is unsafe [-Wunsafe-buffer-usage-in-libc-call-numeric-conversion]
// CHECK-DAG: function 'strcpy' is unsafe [-Wunsafe-buffer-usage-in-libc-call-string-manipulation]
// CHECK-DAG: function 'wcslen' is unsafe [-Wunsafe-buffer-usage-in-libc-call-string-examination]
// CHECK-DAG: function 'memset' is unsafe [-Wunsafe-buffer-usage-in-libc-call-memory]
// CHECK-DAG: function 'bzero' is unsafe [-Wunsafe-buffer-usage-in-libc-call-memory]
// CHECK-DAG: function 'fgets' is unsafe [-Wunsafe-buffer-usage-in-libc-call-io]
// CHECK-DAG: function 'sprintf' is unsafe [-Wunsafe-buffer-usage-in-libc-call-printf]
// CHECK-DAG: function 'sscanf' is unsafe [-Wunsafe-buffer-usage-in-libc-call-scanf]
// CHECK-DAG: function 'qsort' is unsafe [-Wunsafe-buffer-usage-in-libc-call]
