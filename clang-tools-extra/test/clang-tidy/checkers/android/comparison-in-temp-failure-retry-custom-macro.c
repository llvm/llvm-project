// RUN: %check_clang_tidy %s android-comparison-in-temp-failure-retry %t -- -config="{CheckOptions: {android-comparison-in-temp-failure-retry.RetryMacros: 'MY_TEMP_FAILURE_RETRY,MY_OTHER_TEMP_FAILURE_RETRY'}}"

#define MY_TEMP_FAILURE_RETRY(x) \
  ({                             \
    typeof(x) __z;               \
    do                           \
      __z = (x);                 \
    while (__z == -1);           \
    __z;                         \
  })

#define MY_OTHER_TEMP_FAILURE_RETRY(x) \
  ({                                   \
    typeof(x) __z;                     \
    do                                 \
      __z = (x);                       \
    while (__z == -1);                 \
    __z;                               \
  })

int foo(void);
int bar(int a);

void with_custom_macro(void) {
  MY_TEMP_FAILURE_RETRY(foo());
  MY_TEMP_FAILURE_RETRY(foo() == 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: top-level comparison in MY_TEMP_FAILURE_RETRY
  MY_TEMP_FAILURE_RETRY((foo()));
  MY_TEMP_FAILURE_RETRY((int)(foo() == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: top-level comparison in MY_TEMP_FAILURE_RETRY
  MY_TEMP_FAILURE_RETRY((bar(foo() == 1)));
  MY_TEMP_FAILURE_RETRY((int)((bar(foo() == 1)) == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:49: warning: top-level comparison in MY_TEMP_FAILURE_RETRY
}

void with_other_custom_macro(void) {
  MY_OTHER_TEMP_FAILURE_RETRY(foo());
  MY_OTHER_TEMP_FAILURE_RETRY(foo() == 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: top-level comparison in MY_OTHER_TEMP_FAILURE_RETRY
  MY_OTHER_TEMP_FAILURE_RETRY((foo()));
  MY_OTHER_TEMP_FAILURE_RETRY((int)(foo() == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: top-level comparison in MY_OTHER_TEMP_FAILURE_RETRY
  MY_OTHER_TEMP_FAILURE_RETRY((bar(foo() == 1)));
  MY_OTHER_TEMP_FAILURE_RETRY((int)((bar(foo() == 1)) == 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:55: warning: top-level comparison in MY_OTHER_TEMP_FAILURE_RETRY
}
