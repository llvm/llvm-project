// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.unix.Stream -analyzer-store region -verify %s

typedef struct _IO_FILE FILE;
extern FILE *fopen(const char *path, const char *mode);

struct X {
  int A;
  int B;
};

void *fopen(X x, const char *mode) {
  return new char[4];
}

void f1() {
  X X1;
  void *p = fopen(X1, "oo");
} // no-warning

void f2() {
  FILE *f = fopen("file", "r");
} // expected-warning {{Opened File never closed. Potential Resource leak}}
