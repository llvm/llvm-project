// RUN: %clang_analyze_cc1 -Wno-array-bounds -analyzer-store=region -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix \
// RUN:   -analyzer-checker=alpha.security.ArrayBound \
// RUN:   -analyzer-config unix:Optimistic=true

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void *calloc(size_t, size_t);

char f1() {
  char* s = "abcd";
  char c = s[4]; // no-warning
  return s[5] + c; // expected-warning{{Access out-of-bound array element (buffer overflow)}}
}

void f2() {
  int *p = malloc(12);
  p[3] = 4; // expected-warning{{Access out-of-bound array element (buffer overflow)}}
}

struct three_words {
  int c[3];
};

struct seven_words {
  int c[7];
};

void f3() {
  struct three_words a, *p;
  p = &a;
  p[0] = a; // no-warning
  p[1] = a; // expected-warning{{Access out-of-bound array element (buffer overflow)}}
}

void f4() {
  struct seven_words c;
  struct three_words a, *p = (struct three_words *)&c;
  p[0] = a; // no-warning
  p[1] = a; // no-warning
  p[2] = a; // expected-warning{{Access out-of-bound array element (buffer overflow)}}
}

void f5() {
  char *p = calloc(2,2);
  p[3] = '.'; // no-warning
  p[4] = '!'; // expected-warning{{out-of-bound}}
}

void f6() {
  char a[2];
  int *b = (int*)a;
  b[1] = 3; // expected-warning{{out-of-bound}}
}

void f7() {
  struct three_words a;
  a.c[3] = 1; // expected-warning{{out-of-bound}}
}

void vla(int a) {
  if (a == 5) {
    int x[a];
    x[4] = 4; // no-warning
    x[5] = 5; // expected-warning{{out-of-bound}}
  }
}

void alloca_region(int a) {
  if (a == 5) {
    char *x = __builtin_alloca(a);
    x[4] = 4; // no-warning
    x[5] = 5; // expected-warning{{out-of-bound}}
  }
}

int symbolic_index(int a) {
  int x[2] = {1, 2};
  if (a == 2) {
    return x[a]; // expected-warning{{out-of-bound}}
  }
  return 0;
}

int symbolic_index2(int a) {
  int x[2] = {1, 2};
  if (a < 0) {
    return x[a]; // expected-warning{{out-of-bound}}
  }
  return 0;
}

int overflow_binary_search(double in) {
  int eee = 16;
  if (in < 1e-8 || in > 1e23) {
    return 0;
  } else {
    static const double ins[] = {1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
                                 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7,
                                 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
                                 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
    if (in < ins[eee]) {
      eee -= 8;
    } else {
      eee += 8;
    }
    if (in < ins[eee]) {
      eee -= 4;
    } else {
      eee += 4;
    }
    if (in < ins[eee]) {
      eee -= 2;
    } else {
      eee += 2;
    }
    if (in < ins[eee]) {
      eee -= 1;
    } else {
      eee += 1;
    }
    if (in < ins[eee]) { // expected-warning {{Access out-of-bound array element (buffer overflow)}}
      eee -= 1;
    }
  }
  return eee;
}
