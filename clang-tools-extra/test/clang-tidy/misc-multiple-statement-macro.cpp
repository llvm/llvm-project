// RUN: %check_clang_tidy %s misc-multiple-statement-macro %t

void F();

#define BAD_MACRO(x) \
  F();               \
  F()

#define GOOD_MACRO(x) \
  do {                \
    F();              \
    F();              \
  } while (0)

#define GOOD_MACRO2(x) F()

#define GOOD_MACRO3(x) F();

#define MACRO_ARG_MACRO(X) \
  if (54)                  \
  X(2)

#define ALL_IN_MACRO(X) \
  if (43)               \
    F();                \
  F()

#define GOOD_NESTED(x)   \
  if (x)            \
    GOOD_MACRO3(x); \
  F();

#define IF(x) if(x)

void positives() {
  if (1)
    BAD_MACRO(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: multiple statement macro used without braces; some statements will be unconditionally executed [misc-multiple-statement-macro]
  if (1) {
  } else
    BAD_MACRO(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: multiple statement macro used
  while (1)
    BAD_MACRO(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: multiple statement macro used
  for (;;)
    BAD_MACRO(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: multiple statement macro used

  MACRO_ARG_MACRO(BAD_MACRO);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: multiple statement macro used
  MACRO_ARG_MACRO(F(); int);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: multiple statement macro used
  IF(1) BAD_MACRO(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: multiple statement macro used
}

void negatives() {
  if (1) {
    BAD_MACRO(1);
  } else {
    BAD_MACRO(1);
  }
  while (1) {
    BAD_MACRO(1);
  }
  for (;;) {
    BAD_MACRO(1);
  }

  if (1)
    GOOD_MACRO(1);
  if (1) {
    GOOD_MACRO(1);
  }
  if (1)
    GOOD_MACRO2(1);
  if (1)
    GOOD_MACRO3(1);

  MACRO_ARG_MACRO(GOOD_MACRO);
  ALL_IN_MACRO(1);

  IF(1) GOOD_MACRO(1);
}
