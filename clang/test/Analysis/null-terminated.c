// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.NullTerminated -DDEFAULT -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.NullTerminated \
// RUN:   -analyzer-config alpha.core.NullTerminated:MaxArraySize=3 \
// RUN:   -DMAX_ARR -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.NullTerminated \
// RUN:   -analyzer-config region-store-max-binding-fanout=0 \
// RUN:   -DUNLIMITED_FANOUT -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.NullTerminated \
// RUN:   -fstrict-flex-arrays=2 -DSTRICT_FLEX -verify %s

#define NULL_TERMINATED __attribute__((annotate("null_terminated")))

void receive(NULL_TERMINATED const int signals[]);
void receive_after(const int signals[] NULL_TERMINATED);

struct c89_fam {
  int n;
  int data[1];
};

// Not a FAM (not trailing).
struct one_element_first {
  int data[1];
  int n;
};

#define TEN_ONES 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
#define HUNDRED_ONES                                                           \
  TEN_ONES, TEN_ONES, TEN_ONES, TEN_ONES, TEN_ONES, TEN_ONES, TEN_ONES,        \
      TEN_ONES, TEN_ONES, TEN_ONES

#ifdef DEFAULT
void test_static_bad(void) {
  int sigs[] = {1, 2, 3};
  receive(sigs);  // expected-warning{{array argument is not null-terminated}}
}

void test_static_good(void) {
  int sigs[] = {1, 2, 0};
  receive(sigs);
}

void test_attr_after_param(void) {
  int sigs[] = {1, 2, 3};
  receive_after(sigs);  // expected-warning{{array argument is not null-terminated}}
}

void test_imperative_bad(void) {
  int sigs[3];
  sigs[0] = 1;
  sigs[1] = 2;
  sigs[2] = 3;
  receive(sigs);  // expected-warning{{array argument is not null-terminated}}
}

void test_modified_bad(void) {
  int sigs[] = {1, 2, 0};
  sigs[2] = 3;
  receive(sigs);  // expected-warning{{array argument is not null-terminated}}
}

void test_early_term_first(void) {
  int sigs[] = {0, 1, 2};
  receive(sigs);
}

void test_only_null_term(void) {
  int sigs[] = {0};
  receive(sigs);
}

void test_conditional(int cond) {
  int sigs[] = {1, 2, 0};
  if (cond)
    sigs[2] = 3;
  receive(sigs);  // expected-warning{{array argument is not null-terminated}}
}

// Zero-length array.
struct flex {
  int n;
  int data[0];
};
void test_zero_length(struct flex *f) {
  receive(f->data);
}

void receive_char(NULL_TERMINATED const char buf[]);

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

void test_char_no_room_for_term(void) {
  char buf[2] = "hi";
  receive_char(buf);  // expected-warning{{array argument is not null-terminated}}
}

void test_char_string_initializer(void) {
  char buf[] = "hi";
  receive_char(buf);
}

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

// Global arrays are a CSA limitation.
int global_bad[] = {1, 2, 3};
int global_good[] = {1, 2, 0};

void test_global(void) {
  receive(global_bad);
  receive(global_good);
}

void test_static_local_bad(void) {
  static int sigs[] = {1, 2, 3};
  receive(sigs);// expected-warning{{array argument is not null-terminated}}
}

// memset is a CSA limitation.
void *memset(void *, int, __SIZE_TYPE__);

void test_memset_nonzero(void) {
  int sigs[4] = {0, 0, 0, 0};
  memset(sigs, -1, sizeof(sigs)); // set all bits to 1
  receive(sigs);
}

void test_memset_zero(void) {
  int sigs[4] = {1, 2, 3, 4};
  memset(sigs, 0, sizeof(sigs));
  receive(sigs);
}

void test_short_initializer(void) {
  // The other 6 elements are zero-initialized.
  int sigs[8] = {1, 2};
  receive(sigs);
}

void test_empty_initializer(void) {
  int sigs[8] = {};
  receive(sigs);
}

void test_partially_initialized(void) {
  // No initializer list: unwritten elements are undefined (hence unknown to
  // the analyzer)
  int sigs[3];
  sigs[0] = 1;
  sigs[1] = 2;
  receive(sigs);
}

struct with_pad {
  int pad;
  int sigs[3];
};

void test_field_bad(void) {
  struct with_pad s = {9, {1, 2, 3}};
  receive(s.sigs);  // expected-warning{{array argument is not null-terminated}}
}

void test_field_good(void) {
  struct with_pad s = {9, {1, 2, 0}};
  receive(s.sigs);
}

void test_field_of_pointee(struct with_pad *p) {
  receive(p->sigs);
}

struct wrapper {
  int sigs[3];
};

void test_copy_bad(void) {
  struct wrapper w = {{1, 2, 3}};
  struct wrapper v = w;
  receive(v.sigs);  // expected-warning{{array argument is not null-terminated}}
}

void test_copy_good(void) {
  struct wrapper w = {{1, 2, 0}};
  struct wrapper v = w;
  receive(v.sigs);
}

void test_symbolic_index(int i) {
  int sigs[3] = {1, 2, 3};
  sigs[i] = 0;  // i is unknown, write may terminate array.
  receive(sigs);
}

struct two_arrays {
  int a[3];
  int b[3];
};

// Writes to sibling objects should not affect others.
void test_symbolic_index_sibling(int i) {
  struct two_arrays s = {{1, 2, 3}, {4, 5, 6}};
  s.b[i] = 0;
  receive(s.a); // expected-warning{{array argument is not null-terminated}}
}

// Offset unknown: write may have terminated array.
void test_symbolic_offset_in_object(int i) {
  struct two_arrays s = {{1, 2, 3}, {4, 5, 6}};
  *((int *)&s + i) = 0;
  receive(s.a);
}

void test_partial_element_write(void) {
  int sigs[3] = {1, 2, 3};
  *(char *)sigs = 0;  // char written to int: to granular to reason about.
  receive(sigs);
}

// FIXME: a pointer into the middle of the array is checked as the whole array,
// despite the callee never seeing the null terminator, resulting in a false
// negative.
void test_interior_pointer(void) {
  int sigs[3] = {0, 1, 2};
  receive(&sigs[1]);
}

void receive_long(NULL_TERMINATED const long signals[]);

void test_long_bad(void) {
  long sigs[3] = {1, 2, 3};
  receive_long(sigs); // expected-warning{{array argument is not null-terminated}}
}

void receive_ptr(NULL_TERMINATED void *const ptrs[]);

void test_ptr_bad(void) {
  int x, y;
  void *ptrs[2] = {&x, &y};
  receive_ptr(ptrs);  // expected-warning{{array argument is not null-terminated}}
}

void test_ptr_good(void) {
  int x;
  void *ptrs[2] = {&x, 0};
  receive_ptr(ptrs);
}

void test_symbolic_unknown(int x) {
  int sigs[2] = {1, x};
  receive(sigs);
}

void test_symbolic_nonzero(int x) {
  int sigs[2] = {1, x};
  // x is symbolic but guaranteed to be non-zero on this path.
  if (x > 0)
    receive(sigs);  // expected-warning{{array argument is not null-terminated}}
}

void test_symbolic_expression_nonzero(int x) {
  int sigs[2] = {1, x + 5};
  if (x > -5)
    receive(sigs);  // expected-warning{{array argument is not null-terminated}}
}

void test_c89_fam(struct c89_fam *f) {
  f->data[0] = 1;
  receive(f->data);
}

void test_one_element_not_last(void) {
  struct one_element_first s = {{1}, 0};
  receive(s.data);  // expected-warning{{array argument is not null-terminated}}
}

// region-store-max-binding-fanout limits how many elements the analyzer will
// explicitly unpack from a single aggregate initializer, no matter what
// MaxArraySize allows, so the excess contents of the array are unknown.
void test_over_binding_fanout(void) {
  // 200 is greater than the default value of 128.
  int sigs[200] = {HUNDRED_ONES, HUNDRED_ONES};
  receive(sigs);
}

#endif  // DEFAULT

#ifdef UNLIMITED_FANOUT
void test_over_binding_fanout_unlimited(void) {
  int sigs[200] = {HUNDRED_ONES, HUNDRED_ONES};
  receive(sigs);  // expected-warning{{array argument is not null-terminated}}
}

#endif  // UNLIMITED_FANOUT

#ifdef STRICT_FLEX
// -fstrict-flex-arrays of level 2 means we shouldn't treat trailing array of
// one element as a FAM, so we can model it here.
void test_c89_fam_strict(struct c89_fam *f) {
  f->data[0] = 1;
  receive(f->data); // expected-warning{{array argument is not null-terminated}}
}

#endif  // STRICT_FLEX

#ifdef MAX_ARR
void test_maxarraysize_at_limit(void) {
  int sigs[3] = {1, 2, 3};
  receive(sigs);  // expected-warning{{array argument is not null-terminated}}
}

void test_maxarraysize_over_limit(void) {
  int sigs[4] = {1, 2, 3, 4};
  receive(sigs);
}

#endif  // MAX_ARR
