// RUN: %check_clang_tidy -std=c11-or-later %s modernize-macro-to-enum %t-c -- -- -Wno-unused-value
// RUN: %clang -std=c11 -Wno-unused-value -fsyntax-only %t-c.c
// RUN: %check_clang_tidy -std=c++14-or-later -assume-filename=arguments.cpp %s modernize-macro-to-enum %t-cxx -- -- -Wno-unused-value
// RUN: %clang -std=c++14 -Wno-unused-value -fsyntax-only %t-cxx.cpp

void use_one(int);

#define CURLMIN(a, b) ((a) < (b) ? (a) : (b))
#define FORWARD_MIN(a, b) CURLMIN(a, b)

// Each substituted copy is a complete expression, including forwarded copies.
#define CW_PAUSE_DEC_WRITE_CHUNK (4096)
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'CW_PAUSE_DEC_WRITE_CHUNK' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: CW_PAUSE_DEC_WRITE_CHUNK = (4096)
// CHECK-FIXES-NEXT: };

int chunk_size(int length) {
  return CURLMIN(length, CW_PAUSE_DEC_WRITE_CHUNK);
}

int forwarded_chunk_size(int length) {
  return FORWARD_MIN(length, CW_PAUSE_DEC_WRITE_CHUNK);
}

#define SUM_VALUE 1 + 2
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'SUM_VALUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: SUM_VALUE = 1 + 2
// CHECK-FIXES-NEXT: };

int sum_size(int length) {
  return CURLMIN(length, SUM_VALUE);
}

struct hwb_color {
  float H, W, B;
};

#define RETURN_HWB(h, w, b) { HWB->H = h; HWB->W = w; HWB->B = b; return HWB; }

// A constant used in a statement macro still forms a complete expression.
#define HWB_UNDEFINED -1
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'HWB_UNDEFINED' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: HWB_UNDEFINED = -1
// CHECK-FIXES-NEXT: };

struct hwb_color *undefined_hue(struct hwb_color *HWB, float w, float b) {
  RETURN_HWB(HWB_UNDEFINED, w, b);
}

// One complete copy cannot validate another copy used as partial syntax.
#define SAFE_FIRST(x) ((x), 3 * x)
#define UNSAFE_FIRST(x) (3 * x, (x))
#define MIXED_VALUE 1 + 2
// CHECK-FIXES: #define MIXED_VALUE 1 + 2

int mixed_first(void) {
  return SAFE_FIRST(MIXED_VALUE);
}

#define REVERSED_VALUE 1 + 2
// CHECK-FIXES: #define REVERSED_VALUE 1 + 2

int mixed_last(void) {
  return UNSAFE_FIRST(REVERSED_VALUE);
}

#define MIXED_CALL(x) (use_one(x), use_one x)
#define CALL_VALUE (7)
// CHECK-FIXES: #define CALL_VALUE (7)

void mixed_call(void) {
  MIXED_CALL(CALL_VALUE);
}

// A forwarded copy used only to construct a string is not an expression.
#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)
#define USE_AND_STRINGIFY(x) ((void)(x), STRINGIFY(x))
#define STRING_VALUE 42
// CHECK-FIXES: #define STRING_VALUE 42

const char *string_value(void) {
  return USE_AND_STRINGIFY(STRING_VALUE);
}

// A separate valid expansion cannot hide an invalid macro argument.
#define MULTIPLY(x) 3 * x
#define PARTIAL_VALUE 1 + 2
// CHECK-FIXES: #define PARTIAL_VALUE 1 + 2

int Direct = PARTIAL_VALUE;
int Partial = MULTIPLY(PARTIAL_VALUE);
