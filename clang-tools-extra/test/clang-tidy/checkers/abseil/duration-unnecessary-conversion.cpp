// RUN: %check_clang_tidy -std=c++11-or-later %s abseil-duration-unnecessary-conversion %t -- -- -I %S/Inputs

#include "absl/time/time.h"

void f() {
  absl::Duration d1, d2;

  // Floating point
  d2 = absl::Hours(absl::ToDoubleHours(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Minutes(absl::ToDoubleMinutes(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Seconds(absl::ToDoubleSeconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Milliseconds(absl::ToDoubleMilliseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Microseconds(absl::ToDoubleMicroseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Nanoseconds(absl::ToDoubleNanoseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1

  // Integer point
  d2 = absl::Hours(absl::ToInt64Hours(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Minutes(absl::ToInt64Minutes(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Seconds(absl::ToInt64Seconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Milliseconds(absl::ToInt64Milliseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Microseconds(absl::ToInt64Microseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Nanoseconds(absl::ToInt64Nanoseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1

  d2 = absl::Hours(d1 / absl::Hours(1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Minutes(d1 / absl::Minutes(1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Seconds(d1 / absl::Seconds(1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Milliseconds(d1 / absl::Milliseconds(1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Microseconds(d1 / absl::Microseconds(1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Nanoseconds(d1 / absl::Nanoseconds(1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1

  d2 = absl::Hours(absl::FDivDuration(d1, absl::Hours(1)));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Minutes(absl::FDivDuration(d1, absl::Minutes(1)));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Seconds(absl::FDivDuration(d1, absl::Seconds(1)));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Milliseconds(absl::FDivDuration(d1, absl::Milliseconds(1)));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Microseconds(absl::FDivDuration(d1, absl::Microseconds(1)));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1
  d2 = absl::Nanoseconds(absl::FDivDuration(d1, absl::Nanoseconds(1)));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1

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

  // Multiplication
  d2 = absl::Nanoseconds(absl::ToDoubleNanoseconds(d1) * 2);
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1 * 2
  d2 = absl::Microseconds(absl::ToInt64Microseconds(d1) * 2);
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1 * 2
  d2 = absl::Milliseconds(absl::ToDoubleMilliseconds(d1) * 2);
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1 * 2
  d2 = absl::Seconds(absl::ToInt64Seconds(d1) * 2);
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1 * 2
  d2 = absl::Minutes(absl::ToDoubleMinutes(d1) * 2);
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1 * 2
  d2 = absl::Hours(absl::ToInt64Hours(d1) * 2);
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = d1 * 2
  d2 = absl::Nanoseconds(2 * absl::ToDoubleNanoseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = 2 * d1
  d2 = absl::Microseconds(2 * absl::ToInt64Microseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = 2 * d1
  d2 = absl::Milliseconds(2 * absl::ToDoubleMilliseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = 2 * d1
  d2 = absl::Seconds(2 * absl::ToInt64Seconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = 2 * d1
  d2 = absl::Minutes(2 * absl::ToDoubleMinutes(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = 2 * d1
  d2 = absl::Hours(2 * absl::ToInt64Hours(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: remove unnecessary absl::Duration conversions [abseil-duration-unnecessary-conversion]
  // CHECK-FIXES: d2 = 2 * d1

  // These should not match
  d2 = absl::Seconds(absl::ToDoubleMilliseconds(d1));
  d2 = absl::Seconds(4);
  int i = absl::ToInt64Milliseconds(d1);
  d2 = absl::Hours(d1 / absl::Minutes(1));
  d2 = absl::Seconds(d1 / absl::Seconds(30));
  d2 = absl::Hours(absl::FDivDuration(d1, absl::Minutes(1)));
  d2 = absl::Milliseconds(absl::FDivDuration(d1, absl::Milliseconds(20)));
  d2 = absl::Seconds(absl::ToInt64Milliseconds(d1) * 2);
  d2 = absl::Milliseconds(absl::ToDoubleSeconds(d1) * 2);
}
