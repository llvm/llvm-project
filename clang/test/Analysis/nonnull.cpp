// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core -verify %s

void nonnull [[gnu::nonnull]] (int *q);

void f1(int *p) {
  if (p)
    return;
  nonnull(p); //expected-warning{{nonnull}}
}

void f2(int *p) {
  if (p)
    return;
  auto lambda = [](int *q) __attribute__((nonnull)){};
  lambda(p); //expected-warning{{nonnull}}
}

template <class... ARGS>
void variadicNonnull(ARGS... args) __attribute__((nonnull));

void f3(int a, float b, int *p) {
  if (p)
    return;
  variadicNonnull(a, b, p); //expected-warning{{nonnull}}
}

int globalVar = 15;
void moreParamsThanArgs [[gnu::nonnull(2, 4)]] (int a, int *p, int b = 42, int *q = &globalVar);

void f4(int a, int *p) {
  if (p)
    return;
  moreParamsThanArgs(a, p); //expected-warning{{nonnull}}
}
