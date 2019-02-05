// RUN: %check_clang_tidy %s abseil-duration-unnecessary-conversion %t -- -- -I%S/Inputs

#include "absl/time/time.h"

void f() {
  absl::Duration d1, d2;

  // Floating point
  d2 = absl::Hours(absl::ToDoubleHours(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d1
  d2 = absl::Minutes(absl::ToDoubleMinutes(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d1
  d2 = absl::Seconds(absl::ToDoubleSeconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d1
  d2 = absl::Milliseconds(absl::ToDoubleMilliseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d1
  d2 = absl::Microseconds(absl::ToDoubleMicroseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d1
  d2 = absl::Nanoseconds(absl::ToDoubleNanoseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d1

  // Integer point
  d2 = absl::Hours(absl::ToInt64Hours(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d1
  d2 = absl::Minutes(absl::ToInt64Minutes(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d1
  d2 = absl::Seconds(absl::ToInt64Seconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d1
  d2 = absl::Milliseconds(absl::ToInt64Milliseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d1
  d2 = absl::Microseconds(absl::ToInt64Microseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d1
  d2 = absl::Nanoseconds(absl::ToInt64Nanoseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d1

  // As macro argument
#define PLUS_FIVE_S(x) x + absl::Seconds(5)
  d2 = PLUS_FIVE_S(absl::Seconds(absl::ToInt64Seconds(d1)));
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: PLUS_FIVE_S(d1)
#undef PLUS_FIVE_S

  // Split by macro: should not change
#define TOSECONDS(x) absl::Seconds(x)
  d2 = TOSECONDS(absl::ToInt64Seconds(d1));
#undef TOSECONDS

  // Don't change something inside a macro definition
#define VALUE(x) absl::Hours(absl::ToInt64Hours(x));
  d2 = VALUE(d1);
#undef VALUE

  // These should not match
  d2 = absl::Seconds(absl::ToDoubleMilliseconds(d1));
  d2 = absl::Seconds(4);
  int i = absl::ToInt64Milliseconds(d1);
}
