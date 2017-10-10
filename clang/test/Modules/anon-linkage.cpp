// RUN: %clang_cc1 -std=c++17 -fmodules-ts %s

typedef struct {
  int c;
  union {
    int n;
    char c[4];
  } v;
} mbstate;

export module M;
export using ::mbstate;
