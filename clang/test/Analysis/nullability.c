// RUN: %clang_analyze_cc1 -fblocks -analyzer-checker=core,nullability,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached(void);

void it_takes_two(int a, int b);
void function_pointer_arity_mismatch(void) {
  void(*fptr)() = it_takes_two;
  fptr(1); // no-crash expected-warning {{Function taking 2 arguments is called with fewer (1)}}
  // expected-warning@-1 {{passing arguments to a function without a prototype is deprecated in all versions of C and is not supported in C23}}
}

void block_arity_mismatch(void) {
  void(^b)() = ^(int a, int b) { };
  b(1);  // no-crash expected-warning {{Block taking 2 arguments is called with fewer (1)}}
  // expected-warning@-1 {{passing arguments to a function without a prototype is deprecated in all versions of C and is not supported in C23}}
}

int *nonnull_return_annotation_indirect(void) __attribute__((returns_nonnull));
int *nonnull_return_annotation_indirect(void) {
  int *x = 0;
  return x; // expected-warning {{Null returned from a function that is expected to return a non-null value}}
}

int *nonnull_return_annotation_direct(void) __attribute__((returns_nonnull));
int *nonnull_return_annotation_direct(void) {
  return 0; // expected-warning {{Null returned from a function that is expected to return a non-null value}}
} // expected-warning@-1 {{null returned from function that requires a non-null return value}}

int *nonnull_return_annotation_assumed(int* ptr) __attribute__((returns_nonnull));
int *nonnull_return_annotation_assumed(int* ptr) {
  if (ptr) {
    return ptr;
  }
  return ptr; // expected-warning {{Null returned from a function that is expected to return a non-null value}}
}

int *produce_nonnull_ptr(void) __attribute__((returns_nonnull));

__attribute__((returns_nonnull))
int *cannot_return_null(void) {
  int *x = produce_nonnull_ptr();
  if (!x) {
    clang_analyzer_warnIfReached();
    // expected-warning@-1 {{REACHABLE}}
    // TODO: This warning is a false positive, according to the contract of
    // produce_nonnull_ptr, x cannot be null.
  }
  // Regardless of the potential state split above, x cannot be nullptr
  // according to the produce_nonnull_ptr annotation.
  return x;
  // False positive: expected-warning@-1 {{Null returned from a function that is expected to return a non-null value}}
}

__attribute__((returns_nonnull)) int *passthrough(int *p) {
  return p; // no-warning: we have no evidence that `p` is null, i.e., violating the contract
}

__attribute__((returns_nonnull)) int *passthrough2(int *p);
int *passthrough2(int *p) {
  return p; // expected-warning{{Null returned from a function that is expected to return a non-null value}}
}

void call_with_null(void) {
  passthrough2(0);
}
