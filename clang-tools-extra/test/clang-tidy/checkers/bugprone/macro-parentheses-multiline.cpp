// RUN: %check_clang_tidy %s bugprone-macro-parentheses %t
// RUN: %clang_cc1 -fsyntax-only %t.cpp

// Preserve the indentation and line continuations being tested.
// clang-format off
#define UNINDENTED(b0, b1) \
b0=b1; \
b1++
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: macro argument should be enclosed in parentheses [bugprone-macro-parentheses]
// CHECK-FIXES: #define UNINDENTED(b0, b1) \
// CHECK-FIXES-NEXT: b0=b1; \
// CHECK-FIXES-NEXT: (b1)++

#define INDENTED(b0, b1) \
  b0=b1; \
  b1++
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: macro argument should be enclosed in parentheses [bugprone-macro-parentheses]
// CHECK-FIXES: #define INDENTED(b0, b1) \
// CHECK-FIXES-NEXT: b0=b1; \
// CHECK-FIXES-NEXT: (b1)++

#define SPLIT_ARGUMENT(b0, b1) \
b0=b1; \
b\
1++
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: macro argument should be enclosed in parentheses [bugprone-macro-parentheses]
// CHECK-FIXES: #define SPLIT_ARGUMENT(b0, b1) \
// CHECK-FIXES-NEXT: b0=b1; \
// CHECK-FIXES-NEXT: (b\
// CHECK-FIXES-NEXT: 1)++

#define REPEATED_CONTINUATION(b0, b1) \
b0=b1; \
\
b1++
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: macro argument should be enclosed in parentheses [bugprone-macro-parentheses]
// CHECK-FIXES: #define REPEATED_CONTINUATION(b0, b1) \
// CHECK-FIXES-NEXT: b0=b1; \
// CHECK-FIXES-NEXT: \
// CHECK-FIXES-NEXT: (b1)++

#define REPLACEMENT_BEGIN \
1+2
// CHECK-MESSAGES: :[[@LINE-1]]:2: warning: macro replacement list should be enclosed in parentheses [bugprone-macro-parentheses]
// CHECK-FIXES: #define REPLACEMENT_BEGIN \
// CHECK-FIXES-NEXT: (1+2)

#define REPLACEMENT_OPERATOR 1 \
+2
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: macro replacement list should be enclosed in parentheses [bugprone-macro-parentheses]
// CHECK-FIXES: #define REPLACEMENT_OPERATOR (1 \
// CHECK-FIXES-NEXT: +2)

#define SPLIT_REPLACEMENT \
+1\
2
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: macro replacement list should be enclosed in parentheses [bugprone-macro-parentheses]
// CHECK-FIXES: #define SPLIT_REPLACEMENT \
// CHECK-FIXES-NEXT: (+1\
// CHECK-FIXES-NEXT: 2)
// clang-format on

int use_macros(int a, int b) {
  UNINDENTED(a, b);
  INDENTED(a, b);
  SPLIT_ARGUMENT(a, b);
  REPEATED_CONTINUATION(a, b);
  return REPLACEMENT_BEGIN + REPLACEMENT_OPERATOR + SPLIT_REPLACEMENT;
}
