// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection,unix.Malloc -analyzer-config c++-allocator-inlining=true -std=c++11 -verify %s

void clang_analyzer_eval(bool);

typedef __typeof__(sizeof(int)) size_t;

void *malloc(size_t size);

void *operator new(size_t size) throw() {
  void *x = malloc(size);
  if (!x)
    return nullptr;
  return x;
}

void checkNewAndConstructorInlining() {
  int *s = new int;
} // expected-warning {{Potential leak of memory pointed to by 's'}}
