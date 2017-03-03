// RUN: %clang_analyze_cc1 -analyzer-store=region -analyzer-checker=core,unix.Malloc -fblocks -verify %s
// RUN: %clang_analyze_cc1 -analyzer-store=region -analyzer-checker=core,unix.Malloc -fblocks -verify -analyzer-config unix.Malloc:Optimistic=true %s
typedef __typeof(sizeof(int)) size_t;
void free(void *);
void *alloca(size_t);

void t1 () {
  int a[] = { 1 };
  free(a); // expected-warning {{Argument to free() is the address of the local variable 'a', which is not memory allocated by malloc()}}
}

void t2 () {
  int a = 1;
  free(&a); // expected-warning {{Argument to free() is the address of the local variable 'a', which is not memory allocated by malloc()}}
}

void t3 () {
  static int a[] = { 1 };
  free(a); // expected-warning {{Argument to free() is the address of the static variable 'a', which is not memory allocated by malloc()}}
}

void t4 (char *x) {
  free(x); // no-warning
}

void t5 () {
  extern char *ptr();
  free(ptr()); // no-warning
}

void t6 () {
  free((void*)1000); // expected-warning {{Argument to free() is a constant address (1000), which is not memory allocated by malloc()}}
}

void t7 (char **x) {
  free(*x); // no-warning
}

void t8 (char **x) {
  // ugh
  free((*x)+8); // no-warning
}

void t9 () {
label:
  free(&&label); // expected-warning {{Argument to free() is the address of the label 'label', which is not memory allocated by malloc()}}
}

void t10 () {
  free((void*)&t10); // expected-warning {{Argument to free() is the address of the function 't10', which is not memory allocated by malloc()}}
}

void t11 () {
  char *p = (char*)alloca(2);
  free(p); // expected-warning {{Memory allocated by alloca() should not be deallocated}}
}

void t12 () {
  char *p = (char*)__builtin_alloca(2);
  free(p); // expected-warning {{Memory allocated by alloca() should not be deallocated}}
}

void t13 () {
  free(^{return;}); // expected-warning {{Argument to free() is a block, which is not memory allocated by malloc()}}
}

void t14 (char a) {
  free(&a); // expected-warning {{Argument to free() is the address of the parameter 'a', which is not memory allocated by malloc()}}
}

static int someGlobal[2];
void t15 () {
  free(someGlobal); // expected-warning {{Argument to free() is the address of the global variable 'someGlobal', which is not memory allocated by malloc()}}
}

void t16 (char **x, int offset) {
  // Unknown value
  free(x[offset]); // no-warning
}
