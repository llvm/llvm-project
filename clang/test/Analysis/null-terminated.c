// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.NullTerminated -DDEFAULT -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.NullTerminated \
// RUN:   -analyzer-config alpha.core.NullTerminated:MaxArraySize=3 \
// RUN:   -DMAX_ARR -verify %s

void receive(__attribute__((null_terminated)) const int signals[]);

#ifdef DEFAULT
void test_static_bad(void) {
  int sigs[] = {1, 2, 3};
  receive(sigs);	// expected-warning{{array argument is not null-terminated}}
}

void test_static_good(void) {
  int sigs[] = {1, 2, 0};
  receive(sigs);
}

void test_imperative_bad(void) {
  int sigs[3];
  sigs[0] = 1;
  sigs[1] = 2;
  sigs[2] = 3;
  receive(sigs);  // expected-warning{{array argument is not null-terminated}}
}

void test_imperative_good(void) {
  int sigs[3];
  sigs[0] = 1;
  sigs[1] = 2;
  sigs[2] = 0;
  receive(sigs);
}

void test_modified_bad(void) {
  int sigs[] = {1, 2, 0};
  sigs[2] = 3;
  receive(sigs);  // expected-warning{{array argument is not null-terminated}}
}

void test_modified_good(void) {
  int sigs[] = {0, 2, 3};
  sigs[1] = 0;
  receive(sigs);
  sigs[2] = 0;
  receive(sigs);
}

// Early-terminated arrays
void test_early_term_middle(void) {
  int sigs[] = {1, 0, 3};
  receive(sigs);
}

void test_early_term_first(void) {
  int sigs[] = {0, 1, 2};
  receive(sigs);
}

// Single zero element
void test_only_null_term(void) {
  int sigs[] = {0};
  receive(sigs);
}

// Conditional path
void test_conditional(int cond) {
  int sigs[] = {1, 2, 0};
  if (cond)
    sigs[2] = 3;
  receive(sigs);  // expected-warning{{array argument is not null-terminated}}
}

// Zero-length array
struct flex {
  int n;
  int data[0];
};
void test_zero_length(struct flex *f) {
  receive(f->data);
}

void receive_char(__attribute__((null_terminated)) const char buf[]);

void test_char_bad(void) {
  char buf[] = {'a', 'b', 'c'};
  receive_char(buf);  // expected-warning{{array argument is not null-terminated}}
}

void test_char_good(void) {
  char buf[] = {'a', '\0', 'c'};
  receive_char(buf);
  receive_char("hello");
  receive_char("");
}

// Compound literals
void test_compound_literal_bad(void) {
  const int *p = (int[]){1, 2, 3};
  receive(p); // expected-warning{{array argument is not null-terminated}}
  receive((int[]){1, 2, 3});  // expected-warning{{array argument is not null-terminated}}
}
void test_compound_literal_direct_good(void) {
  const int *p = (int[]){1, 2, 0};
  receive(p);
  receive((int[]){1, 2, 0});
}

// Global arrays: CSA limitation.
int global_bad[] = {1, 2, 3};
int global_good[] = {1, 2, 0};

void test_global(void) {
  receive(global_bad);
  receive(global_good);
}

// Static local arrays
void test_static_local_bad(void) {
  static int sigs[] = {1, 2, 3};
  receive(sigs);  // expected-warning{{array argument is not null-terminated}}
}

void test_static_local_good(void) {
  static const int sigs[] = {1, 2, 0};
  receive(sigs);
}

// CSA cannot reason about memset - we test here for false positives.
void *memset(void *, int, unsigned long);

void test_memset_zero_bad(void) {
  int sigs[4] = {0, 0, 0, 0};
  memset(sigs, -1, sizeof(sigs)); // set all bits to 1
  receive(sigs);
}

void test_memset_zero_good(void) {
  int sigs[4] = {1, 2, 3, 4};
  memset(sigs, 0, sizeof(sigs));
  receive(sigs);
}

#endif // DEFAULT

#ifdef MAX_ARR
void test_maxarraysize_at_limit(void) {
  int sigs[3] = {1, 2, 3};
  receive(sigs); // expected-warning{{array argument is not null-terminated}}
}

void test_maxarraysize_over_limit(void) {
  int sigs[4] = {1, 2, 3, 4};
  receive(sigs); // no-warning
}

#endif // MAX_ARR
