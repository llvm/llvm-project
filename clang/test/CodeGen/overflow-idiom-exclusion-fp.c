// Check for potential false positives from patterns that _almost_ match classic overflow idioms
// RUN: %clang %s -O2 -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-overflow-pattern-exclusion=all -S -emit-llvm -o - | FileCheck %s
// RUN: %clang %s -O2 -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-overflow-pattern-exclusion=all -fwrapv -S -emit-llvm -o - | FileCheck %s

extern unsigned a, b, c;
extern int u, v, w;

extern unsigned some(void);

// Make sure all these still have handler paths, we shouldn't be excluding
// instrumentation of any "near" idioms.
void close_but_not_quite(void) {
  // CHECK: br i1{{.*}}handler.
  if (a + b > a)
    c = 9;

  // CHECK: br i1{{.*}}handler.
  if (a - b < a)
    c = 9;

  // CHECK: br i1{{.*}}handler.
  if (a + b < a)
    c = 9;

  // CHECK: br i1{{.*}}handler.
  if (a + b + 1 < a)
    c = 9;

  // CHECK: br i1{{.*}}handler.
  if (a + b < a + 1)
    c = 9;

  // CHECK: br i1{{.*}}handler.
  if (b >= a + b)
    c = 9;

  // CHECK: br i1{{.*}}handler.
  if (a + a < a)
    c = 9;

  // CHECK: br i1{{.*}}handler.
  if (a + b == a)
    c = 9;

  // CHECK: br i1{{.*}}handler
  if (u + v < u) /* matches overflow idiom, but is signed */
    c = 9;

  // CHECK: br i1{{.*}}handler
  // Although this can never actually overflow we are still checking that the
  // sanitizer instruments it.
  while (--a)
    some();
}

// cvise'd kernel code that caused problems during development
// CHECK: br i1{{.*}}handler
typedef unsigned size_t;
typedef enum { FSE_repeat_none } FSE_repeat;
typedef enum { ZSTD_defaultAllowed } ZSTD_defaultPolicy_e;
FSE_repeat ZSTD_selectEncodingType_repeatMode;
ZSTD_defaultPolicy_e ZSTD_selectEncodingType_isDefaultAllowed;
size_t ZSTD_NCountCost(void);

void ZSTD_selectEncodingType(void) {
  size_t basicCost =
             ZSTD_selectEncodingType_isDefaultAllowed ? ZSTD_NCountCost() : 0,
         compressedCost = 3 + ZSTD_NCountCost();
  if (basicCost <= compressedCost)
    ZSTD_selectEncodingType_repeatMode = FSE_repeat_none;
}

void function_calls(void) {
  // CHECK: br i1{{.*}}handler
  if (some() + b < some())
    c = 9;
}
