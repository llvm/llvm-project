// RUN: %clang_analyze_cc1 -analyzer-checker=core -w -verify %s

@interface MyObject
- (void)takePointer:(void *)ptr __attribute__((nonnull(1)));
- (void)takePointerArg:(void *)__attribute__((nonnull)) ptr;

@end

void testNonNullMethod(int *p, MyObject *obj) {
  if (p)
    return;
  [obj takePointer:p]; // expected-warning{{nonnull}}
}


@interface Subclass : MyObject
// [[nonnull]] is an inherited attribute.
- (void)takePointer:(void *)ptr;
@end

void testSubclass(int *p, Subclass *obj) {
  if (p)
    return;
  [obj takePointer:p]; // expected-warning{{nonnull}}
}

void testSubclassArg(int *p, Subclass *obj) {
  if (p)
    return;
  [obj takePointerArg:p]; // expected-warning{{nonnull}}
}


union rdar16153464_const_cp_t {
  const struct rdar16153464_cczp *zp;
  const struct rdar16153464_cczp_prime *prime;
} __attribute__((transparent_union));

struct rdar16153464_header {
  union rdar16153464_const_cp_t cp;
  unsigned char pad[16 - sizeof(union rdar16153464_const_cp_t *)];
} __attribute__((aligned(16)));


struct rdar16153464_full_ctx {
  struct rdar16153464_header hdr;
} __attribute__((aligned(16)));


struct rdar16153464_pub_ctx {
  struct rdar16153464_header hdr;
} __attribute__((aligned(16)));


union rdar16153464_full_ctx_t {
  struct rdar16153464_full_ctx *_full;
  struct rdar16153464_header *hdr;
  struct rdar16153464_body *body;
  struct rdar16153464_public *pub;
} __attribute__((transparent_union));

union rdar16153464_pub_ctx_t {
  struct rdar16153464_pub_ctx *_pub;
  struct rdar16153464_full_ctx *_full;
  struct rdar16153464_header *hdr;
  struct rdar16153464_body *body;
  struct rdar16153464_public *pub;
  union rdar16153464_full_ctx_t innert;
} __attribute__((transparent_union));

int rdar16153464(union rdar16153464_full_ctx_t inner)
{
  extern void rdar16153464_check(union rdar16153464_pub_ctx_t outer) __attribute((nonnull(1)));
  rdar16153464_check((union rdar16153464_pub_ctx_t){ .innert = inner }); // no-warning
  rdar16153464_check(inner); // no-warning
  rdar16153464_check(0); // expected-warning{{nonnull}}
}

// Multiple attributes, the basic case
void multipleAttributes_1(char *p, char *q) __attribute((nonnull(1))) __attribute((nonnull(2)));

void testMultiple_1(void) {
  char c;
  multipleAttributes_1(&c, &c); // no-warning
}

void testMultiple_2(void) {
  char c;
  multipleAttributes_1(0, &c); // expected-warning{{nonnull}}
}

void testMultiple_3(void) {
  char c;
  multipleAttributes_1(&c, 0); // expected-warning{{nonnull}}
}

void testMultiple_4(void) {
  multipleAttributes_1(0, 0);// expected-warning{{nonnull}}
}

// Multiple attributes, multiple prototypes
void multipleAttributes_2(char *p, char *q) __attribute((nonnull(1)));
void multipleAttributes_2(char *p, char *q) __attribute((nonnull(2)));

void testMultiple_5(void) {
  char c;
  multipleAttributes_2(0, &c);// expected-warning{{nonnull}}
}

void testMultiple_6(void) {
  char c;
  multipleAttributes_2(&c, 0);// expected-warning{{nonnull}}
}

void testMultiple_7(void) {
  multipleAttributes_2(0, 0);// expected-warning{{nonnull}}
}

// Multiple attributes, same index
void multipleAttributes_3(char *p, char *q) __attribute((nonnull(1))) __attribute((nonnull(1)));

void testMultiple_8(void) {
  char c;
  multipleAttributes_3(0, &c); // expected-warning{{nonnull}}
}

void testMultiple_9(void) {
  char c;
  multipleAttributes_3(&c, 0); // no-warning
}

// Multiple attributes, the middle argument is missing an attribute
void multipleAttributes_4(char *p, char *q, char *r) __attribute((nonnull(1))) __attribute((nonnull(3)));

void testMultiple_10(void) {
  char c;
  multipleAttributes_4(0, &c, &c); // expected-warning{{nonnull}}
}

void testMultiple_11(void) {
  char c;
  multipleAttributes_4(&c, 0, &c); // no-warning
}

void testMultiple_12(void) {
  char c;
  multipleAttributes_4(&c, &c, 0); // expected-warning{{nonnull}}
}


// Multiple attributes, when the last is without index
void multipleAttributes_all_1(char *p, char *q) __attribute((nonnull(1))) __attribute((nonnull));

void testMultiple_13(void) {
  char c;
  multipleAttributes_all_1(0, &c); // expected-warning{{nonnull}}
}

void testMultiple_14(void) {
  char c;
  multipleAttributes_all_1(&c, 0); // expected-warning{{nonnull}}
}

// Multiple attributes, when the first is without index
void multipleAttributes_all_2(char *p, char *q) __attribute((nonnull)) __attribute((nonnull(2)));

void testMultiple_15(void) {
  char c;
  multipleAttributes_all_2(0, &c); // expected-warning{{nonnull}}
}

void testMultiple_16(void) {
  char c;
  multipleAttributes_all_2(&c, 0); // expected-warning{{nonnull}}
}

void testVararg(int k, void *p) {
  extern void testVararg_check(int, ...) __attribute__((nonnull));
  void *n = 0;
  testVararg_check(0);
  testVararg_check(1, p);
  if (k == 1)
    testVararg_check(1, n); // expected-warning{{nonnull}}
  testVararg_check(2, p, p);
  if (k == 2)
    testVararg_check(2, n, p); // expected-warning{{nonnull}}
  if (k == 3)
    testVararg_check(2, p, n); // expected-warning{{nonnull}}
}

void testNotPtr() {
  struct S { int a; int b; int c; } s = {};
  extern void testNotPtr_check(struct S, int) __attribute__((nonnull(1, 2)));
  testNotPtr_check(s, 0);
}
