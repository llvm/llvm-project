// Check for potential false positives from patterns that _almost_ match classic overflow-dependent or overflow-prone code patterns
// RUN: %clang %s -O2 -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-overflow-pattern-exclusion=all -S -emit-llvm -o - | FileCheck %s
// RUN: %clang %s -O2 -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-overflow-pattern-exclusion=all -fwrapv -S -emit-llvm -o - | FileCheck %s

extern unsigned a, b, c;
extern int u, v, w;

extern unsigned some(void);

// Make sure all these still have handler paths, we shouldn't be excluding
// instrumentation of any "near" patterns.
// CHECK-LABEL: close_but_not_quite
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
  // Although this can never actually overflow we are still checking that the
  // sanitizer instruments it.
  while (--a)
    some();
}

// cvise'd kernel code that caused problems during development
typedef unsigned _size_t;
typedef enum { FSE_repeat_none } FSE_repeat;
typedef enum { ZSTD_defaultAllowed } ZSTD_defaultPolicy_e;
FSE_repeat ZSTD_selectEncodingType_repeatMode;
ZSTD_defaultPolicy_e ZSTD_selectEncodingType_isDefaultAllowed;
_size_t ZSTD_NCountCost(void);

// CHECK-LABEL: ZSTD_selectEncodingType
// CHECK: br i1{{.*}}handler
void ZSTD_selectEncodingType(void) {
  _size_t basicCost =
             ZSTD_selectEncodingType_isDefaultAllowed ? ZSTD_NCountCost() : 0,
         compressedCost = 3 + ZSTD_NCountCost();
  if (basicCost <= compressedCost)
    ZSTD_selectEncodingType_repeatMode = FSE_repeat_none;
}

// CHECK-LABEL: function_calls
void function_calls(void) {
  // CHECK: br i1{{.*}}handler
  if (some() + b < some())
    c = 9;
}

// CHECK-LABEL: not_quite_a_negated_unsigned_const
void not_quite_a_negated_unsigned_const(void) {
  // CHECK: br i1{{.*}}handler
  a = -b;
}
