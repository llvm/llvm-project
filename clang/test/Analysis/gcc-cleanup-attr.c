// RUN: %clang_analyze_cc1 -verify %s -Wno-int-to-void-pointer-cast \
// RUN:   -analyzer-checker=core,unix,debug.ExprInspection

void clang_analyzer_dump(const void *);

void defined_cleanup(int *p) {
  clang_analyzer_dump(p); // expected-warning {{&x}}
                          // expected-warning@-1 {{&y}}
  clang_analyzer_dump((const void *)*p); // expected-warning {{42}}
                                         // expected-warning@-1 {{43}}
}

void vardecl_defined() {
  {
    int x __attribute__((cleanup(defined_cleanup)));
    x = 42;
  }
  int y __attribute__((cleanup(defined_cleanup)));
  y = 43;
}

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);

// If cleanup function is not defined, fallback to conservative.
void declared_cleanup(int **p);

void vardecl_declared() {
  int *p1 __attribute__((cleanup(declared_cleanup))) = malloc(sizeof(int));
  int *p2 __attribute__((cleanup(declared_cleanup))) = malloc(sizeof(int));
  *p1 = 42;
  *p2 = 43;
}

void malloc_cleanup(int **p) {
  clang_analyzer_dump((const void *)**p); // expected-warning {{42}}
                                          // expected-warning@-1 {{43}}
  free(*p);
  clang_analyzer_dump(p); // expected-warning {{&p1}}
                          // expected-warning@-1 {{&p2}}
}

void vardecl_malloc() {
  int *p1 __attribute__((cleanup(malloc_cleanup))) = malloc(sizeof(int));
  int *p2 __attribute__((cleanup(malloc_cleanup))) = malloc(sizeof(int));
  *p1 = 42;
  *p2 = 43;
}
