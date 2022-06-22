// RUN: %check_clang_tidy %s abseil-time-comparison %t -- -- -I%S/Inputs

#include "absl/time/time.h"

void f() {
  double x;
  absl::Duration d1, d2;
  bool b;
  absl::Time t1, t2;

  // Check against the RHS
  b = x > absl::ToUnixSeconds(t1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: absl::FromUnixSeconds(x) > t1;
  b = x >= absl::ToUnixSeconds(t1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: absl::FromUnixSeconds(x) >= t1;
  b = x == absl::ToUnixSeconds(t1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: absl::FromUnixSeconds(x) == t1;
  b = x <= absl::ToUnixSeconds(t1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: absl::FromUnixSeconds(x) <= t1;
  b = x < absl::ToUnixSeconds(t1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: absl::FromUnixSeconds(x) < t1;
  b = x == absl::ToUnixSeconds(t1 - d2);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: absl::FromUnixSeconds(x) == t1 - d2;
  b = absl::ToUnixSeconds(t1) > absl::ToUnixSeconds(t2);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: t1 > t2;

  // Check against the LHS
  b = absl::ToUnixSeconds(t1) < x;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: t1 < absl::FromUnixSeconds(x);
  b = absl::ToUnixSeconds(t1) <= x;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: t1 <= absl::FromUnixSeconds(x);
  b = absl::ToUnixSeconds(t1) == x;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: t1 == absl::FromUnixSeconds(x);
  b = absl::ToUnixSeconds(t1) >= x;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: t1 >= absl::FromUnixSeconds(x);
  b = absl::ToUnixSeconds(t1) > x;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: t1 > absl::FromUnixSeconds(x);

  // Comparison against zero
  b = absl::ToUnixSeconds(t1) < 0.0;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: t1 < absl::UnixEpoch();
  b = absl::ToUnixSeconds(t1) < 0;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: t1 < absl::UnixEpoch();

  // Scales other than Seconds
  b = x > absl::ToUnixMicros(t1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: absl::FromUnixMicros(x) > t1;
  b = x >= absl::ToUnixMillis(t1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: absl::FromUnixMillis(x) >= t1;
  b = x == absl::ToUnixNanos(t1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: absl::FromUnixNanos(x) == t1;
  b = x <= absl::ToUnixMinutes(t1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: absl::FromUnixMinutes(x) <= t1;
  b = x < absl::ToUnixHours(t1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: absl::FromUnixHours(x) < t1;

  // A long expression
  bool some_condition;
  int very_very_very_very_long_variable_name;
  absl::Time SomeTime;
  if (some_condition && very_very_very_very_long_variable_name
     < absl::ToUnixSeconds(SomeTime)) {
  // CHECK-MESSAGES: [[@LINE-2]]:25: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: if (some_condition && absl::FromUnixSeconds(very_very_very_very_long_variable_name) < SomeTime) {
    return;
  }

  // A complex expression
  int y;
  b = (y + 5) * 10 > absl::ToUnixMillis(t1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: absl::FromUnixMillis((y + 5) * 10) > t1;

  // We should still transform the expression inside this macro invocation
#define VALUE_IF(v, e) v ? (e) : 0
  int a = VALUE_IF(1, 5 > absl::ToUnixSeconds(t1));
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: VALUE_IF(1, absl::FromUnixSeconds(5) > t1);
#undef VALUE_IF

#define VALUE_IF_2(e) (e)
#define VALUE_IF(v, e) v ? VALUE_IF_2(e) : VALUE_IF_2(0)
  int a2 = VALUE_IF(1, 5 > absl::ToUnixSeconds(t1));
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: perform comparison in the time domain [abseil-time-comparison]
  // CHECK-FIXES: VALUE_IF(1, absl::FromUnixSeconds(5) > t1);
#undef VALUE_IF
#undef VALUE_IF_2

#define VALUE_IF_2(e) (e)
#define VALUE_IF(v, e, type) (v ? VALUE_IF_2(absl::To##type##Seconds(e)) : 0)
  int a3 = VALUE_IF(1, t1, Unix);
#undef VALUE_IF
#undef VALUE_IF_2

#define VALUE_IF_2(e) (e)
#define VALUE_IF(v, e, type) (v ? (5 > VALUE_IF_2(absl::To##type##Seconds(e))) : 0)
  int a4 = VALUE_IF(1, t1, Unix);
#undef VALUE_IF
#undef VALUE_IF_2

  // These should not match
  b = 6 < 4;

#define TODOUBLE(x) absl::ToUnixSeconds(x)
  b = 5.0 > TODOUBLE(t1);
#undef TODOUBLE
#define THIRTY 30.0
  b = THIRTY > absl::ToUnixSeconds(t1);
#undef THIRTY
}
