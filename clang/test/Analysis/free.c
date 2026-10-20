// RUN: %clang_analyze_cc1 -fblocks -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.Malloc
//
// RUN: %clang_analyze_cc1 -fblocks -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-config unix.DynamicMemoryModeling:Optimistic=true
typedef __typeof(sizeof(int)) size_t;
void free(void *);
void *alloca(size_t);

void t1 (void) {
  int a[] = { 1 };
  free(a);
  // expected-warning@-1{{Argument to 'free()' is the address of the local variable 'a', which is not memory allocated by 'malloc()'}}
  // expected-warning@-2{{attempt to call free on non-heap object 'a'}}
}

void t2 (void) {
  int a = 1;
  free(&a);
  // expected-warning@-1{{Argument to 'free()' is the address of the local variable 'a', which is not memory allocated by 'malloc()'}}
  // expected-warning@-2{{attempt to call free on non-heap object 'a'}}
}

void t3 (void) {
  static int a[] = { 1 };
  free(a);
  // expected-warning@-1{{Argument to 'free()' is the address of the static variable 'a', which is not memory allocated by 'malloc()'}}
  // expected-warning@-2{{attempt to call free on non-heap object 'a'}}
}

void t4 (char *x) {
  free(x); // no-warning
}

void t5 (void) {
  extern char *ptr(void);
  free(ptr()); // no-warning
}

void t6 (void) {
  free((void*)1000);
  // expected-warning@-1{{Argument to 'free()' is a constant address (1000), which is not memory allocated by 'malloc()'}}
  // expected-warning@-2{{attempt to call free on non-heap object '(void *)1000'}}
}

void t7 (char **x) {
  free(*x); // no-warning
}

void t8 (char **x) {
  // ugh
  free((*x)+8); // no-warning
}

void t9 (void) {
label:
  free(&&label);
  // expected-warning@-1{{Argument to 'free()' is the address of the label 'label', which is not memory allocated by 'malloc()'}}
  // expected-warning@-2{{attempt to call free on non-heap object 'label'}}
}

void t10 (void) {
  free((void*)&t10);
  // expected-warning@-1{{Argument to 'free()' is the address of the function 't10', which is not memory allocated by 'malloc()'}}
  // expected-warning@-2{{attempt to call free on non-heap object 't10'}}
}

void t11 (void) {
  char *p = (char*)alloca(2);
  free(p); // expected-warning {{Memory allocated by 'alloca()' should not be deallocated}}
}

void t12 (void) {
  char *p = (char*)__builtin_alloca(2);
  free(p); // expected-warning {{Memory allocated by 'alloca()' should not be deallocated}}
}

void t13 (void) {
  free(^{return;});
  // expected-warning@-1{{Argument to 'free()' is a block, which is not memory allocated by 'malloc()'}}
  // expected-warning@-2{{attempt to call free on non-heap object: block expression}}
}

void t14 (char a) {
  free(&a);
  // expected-warning@-1{{Argument to 'free()' is the address of the parameter 'a', which is not memory allocated by 'malloc()'}}
  // expected-warning@-2{{attempt to call free on non-heap object 'a'}}
}

static int someGlobal[2];
void t15 (void) {
  free(someGlobal);
  // expected-warning@-1{{Argument to 'free()' is the address of the global variable 'someGlobal', which is not memory allocated by 'malloc()'}}
  // expected-warning@-2{{attempt to call free on non-heap object 'someGlobal'}}
}

void t16 (char **x, int offset) {
  // Unknown value
  free(x[offset]); // no-warning
}

int *iptr(void);
void t17(void) {
  free(iptr); // Oops, forgot to call iptr().
  // expected-warning@-1{{Argument to 'free()' is the address of the function 'iptr', which is not memory allocated by 'malloc()'}}
  // expected-warning@-2{{attempt to call free on non-heap object 'iptr'}}
}

struct S {
  const char* p;
};

void t18 (struct S s) {
  free((void*)(unsigned long long)s.p); // no warning
}
