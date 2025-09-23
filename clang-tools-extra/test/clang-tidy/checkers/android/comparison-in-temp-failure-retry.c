// RUN: %check_clang_tidy %s android-comparison-in-temp-failure-retry %t

#define TEMP_FAILURE_RETRY(x)                                                  \
  ({                                                                           \
    __typeof__(x) __z;                                                         \
    do                                                                         \
      __z = (x);                                                               \
    while (__z == -1);                                                         \
    __z;                                                                       \
  })

int foo(void);
int bar(int a);

void test(void) {
  int i;
  TEMP_FAILURE_RETRY((i = foo()));
  TEMP_FAILURE_RETRY(foo());
  TEMP_FAILURE_RETRY((foo()));

  TEMP_FAILURE_RETRY(foo() == 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: top-level comparison in TEMP_FAILURE_RETRY [android-comparison-in-temp-failure-retry]
  TEMP_FAILURE_RETRY((foo() == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: top-level comparison in TEMP_FAILURE_RETRY
  TEMP_FAILURE_RETRY((int)(foo() == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: top-level comparison in TEMP_FAILURE_RETRY

  TEMP_FAILURE_RETRY(bar(foo() == 1));
  TEMP_FAILURE_RETRY((bar(foo() == 1)));
  TEMP_FAILURE_RETRY((bar(foo() == 1)) == 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: top-level comparison in TEMP_FAILURE_RETRY
  TEMP_FAILURE_RETRY(((bar(foo() == 1)) == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: top-level comparison in TEMP_FAILURE_RETRY
  TEMP_FAILURE_RETRY((int)((bar(foo() == 1)) == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: top-level comparison in TEMP_FAILURE_RETRY

#define INDIRECT TEMP_FAILURE_RETRY
  INDIRECT(foo());
  INDIRECT((foo() == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: top-level comparison in TEMP_FAILURE_RETRY
  INDIRECT(bar(foo() == 1));
  INDIRECT((int)((bar(foo() == 1)) == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: top-level comparison in TEMP_FAILURE_RETRY

#define TFR(x) TEMP_FAILURE_RETRY(x)
  TFR(foo());
  TFR((foo() == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: top-level comparison in TEMP_FAILURE_RETRY
  TFR(bar(foo() == 1));
  TFR((int)((bar(foo() == 1)) == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: top-level comparison in TEMP_FAILURE_RETRY

#define ADD_TFR(x) (1 + TEMP_FAILURE_RETRY(x) + 1)
  ADD_TFR(foo());
  ADD_TFR(foo() == 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: top-level comparison in TEMP_FAILURE_RETRY

  ADD_TFR(bar(foo() == 1));
  ADD_TFR((int)((bar(foo() == 1)) == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: top-level comparison in TEMP_FAILURE_RETRY

#define ADDP_TFR(x) (1 + TEMP_FAILURE_RETRY((x)) + 1)
  ADDP_TFR(foo());
  ADDP_TFR((foo() == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: top-level comparison in TEMP_FAILURE_RETRY

  ADDP_TFR(bar(foo() == 1));
  ADDP_TFR((int)((bar(foo() == 1)) == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: top-level comparison in TEMP_FAILURE_RETRY

#define MACRO TEMP_FAILURE_RETRY(foo() == 1)
  MACRO;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: top-level comparison in TEMP_FAILURE_RETRY

  // Be sure that being a macro arg doesn't mess with this.
#define ID(x) (x)
  ID(ADDP_TFR(bar(foo() == 1)));
  ID(ADDP_TFR(bar(foo() == 1) == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: top-level comparison in TEMP_FAILURE_RETRY
  ID(MACRO);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: top-level comparison in TEMP_FAILURE_RETRY

#define CMP(x) x == 1
  TEMP_FAILURE_RETRY(CMP(foo()));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: top-level comparison in TEMP_FAILURE_RETRY
}

// Be sure that it works inside of things like loops, if statements, etc.
void control_flow(void) {
  do {
    if (TEMP_FAILURE_RETRY(foo())) {
    }

    if (TEMP_FAILURE_RETRY(foo() == 1)) {
      // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: top-level comparison in TEMP_FAILURE_RETRY
    }

    if (TEMP_FAILURE_RETRY(bar(foo() == 1))) {
    }

    if (TEMP_FAILURE_RETRY(bar(foo() == 1) == 1)) {
      // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: top-level comparison in TEMP_FAILURE_RETRY
    }
  } while (TEMP_FAILURE_RETRY(foo() == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: top-level comparison in TEMP_FAILURE_RETRY
}

void with_nondependent_variable_type(void) {
#undef TEMP_FAILURE_RETRY
#define TEMP_FAILURE_RETRY(x)                                                  \
  ({                                                                           \
    long int __z;                                                              \
    do                                                                         \
      __z = (x);                                                               \
    while (__z == -1);                                                         \
    __z;                                                                       \
  })

  TEMP_FAILURE_RETRY((foo()));
  TEMP_FAILURE_RETRY((int)(foo() == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: top-level comparison in TEMP_FAILURE_RETRY
  TEMP_FAILURE_RETRY((bar(foo() == 1)));
  TEMP_FAILURE_RETRY((int)((bar(foo() == 1)) == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: top-level comparison in TEMP_FAILURE_RETRY
}

// I can't find a case where TEMP_FAILURE_RETRY is implemented like this, but if
// we can cheaply support it, I don't see why not.
void obscured_temp_failure_retry(void) {
#undef TEMP_FAILURE_RETRY
#define IMPL(x)                                                                \
  ({                                                                           \
    __typeof__(x) __z;                                                         \
    do                                                                         \
      __z = (x);                                                               \
    while (__z == -1);                                                         \
    __z;                                                                       \
  })

#define IMPL2(x) IMPL(x)
#define TEMP_FAILURE_RETRY(x) IMPL2(x)
  TEMP_FAILURE_RETRY((foo()));
  TEMP_FAILURE_RETRY((int)(foo() == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: top-level comparison in TEMP_FAILURE_RETRY
  TEMP_FAILURE_RETRY((bar(foo() == 1)));
  TEMP_FAILURE_RETRY((int)((bar(foo() == 1)) == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: top-level comparison in TEMP_FAILURE_RETRY
}
