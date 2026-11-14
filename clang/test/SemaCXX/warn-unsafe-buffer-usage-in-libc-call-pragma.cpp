// RUN: %clang_cc1 -std=c++20 -verify %s

// The analysis is skipped for a function when none of its warnings is
// enabled, so changing a libc-call subgroup with a pragma between functions
// has to change what runs.

typedef __SIZE_TYPE__ size_t;

void *memcpy(void *, const void *, size_t);

void before(void *p, const void *q, size_t n) {
  memcpy(p, q, n);
}

#pragma clang diagnostic warning "-Wunsafe-buffer-usage-in-libc-call-memory"

void enabled(void *p, const void *q, size_t n) {
  memcpy(p, q, n); // expected-warning{{function 'memcpy' is unsafe}}
}

// A function is checked after the pragmas that follow it are read, so this
// declaration makes ignored() the first function checked once they apply.
void ignored(void *p, const void *q, size_t n);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-libc-call-memory"

void ignored(void *p, const void *q, size_t n) {
  memcpy(p, q, n);
}

#pragma clang diagnostic pop

void enabled_again(void *p, const void *q, size_t n) {
  memcpy(p, q, n); // expected-warning{{function 'memcpy' is unsafe}}
}
