// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc -verify %s

typedef __typeof(sizeof(int)) size_t;
void* malloc(size_t size);
void *calloc(size_t num, size_t size);
void free(void * ptr);

void escape(void *);
void next_statement();

void conditional_malloc(bool coin) {
  static int *p;

  if (coin) {
    p = (int *)malloc(sizeof(int));
  }
  p = 0; // Pointee of 'p' dies, which is recognized at the next statement.
  next_statement(); // expected-warning {{Potential memory leak}}
}

void malloc_twice() {
  static int *p;
  p = (int *)malloc(sizeof(int));
  next_statement();
  p = (int *)malloc(sizeof(int));
  next_statement(); // expected-warning {{Potential memory leak}}
  p = 0;
  next_statement(); // expected-warning {{Potential memory leak}}
}

void malloc_escape() {
  static int *p;
  p = (int *)malloc(sizeof(int));
  escape(p); // no-leak
  p = 0; // no-leak
}

void free_whatever_escaped();
void malloc_escape_reversed() {
  static int *p;
  escape(&p);
  p = (int *)malloc(sizeof(int));
  free_whatever_escaped();
  p = 0; // FIXME: We should not report a leak here.
  next_statement(); // expected-warning {{Potential memory leak}}
}

int *malloc_return_static() {
  static int *p = (int *)malloc(sizeof(int));
  return p; // no-leak
}

int malloc_unreachable(int rng) {
  // 'p' does not escape and never freed :(
  static int *p;

  // For the second invocation of this function, we leak the previous pointer.
  // FIXME: We should catch this at some point.
  p = (int *)malloc(sizeof(int));
  *p = 0;

  if (rng > 0)
    *p = rng;

  return *p; // FIXME: We just leaked 'p'. We should warn about this.
}

void malloc_cond(bool cond) {
  static int *p;
  if (cond) {
    p = (int*)malloc(sizeof(int));
    free_whatever_escaped();
    p = 0; // FIXME: We should not report a leak here.
    next_statement(); // expected-warning {{Potential memory leak}}
  }
  escape(&p);
}
