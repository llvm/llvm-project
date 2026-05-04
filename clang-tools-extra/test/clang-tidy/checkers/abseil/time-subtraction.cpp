// RUN: %check_clang_tidy -std=c++11-or-later %s abseil-time-subtraction %t -- -- -I %S/Inputs

#include "absl/time/time.h"

void g(absl::Duration d);

void f() {
  absl::Time t;
  int x, y;
  absl::Duration d;

  d = absl::Hours(absl::ToUnixHours(t) - x);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: d = (t - absl::FromUnixHours(x));
  d = absl::Minutes(absl::ToUnixMinutes(t) - x);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: d = (t - absl::FromUnixMinutes(x));
  d = absl::Seconds(absl::ToUnixSeconds(t) - x);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: d = (t - absl::FromUnixSeconds(x));
  d = absl::Milliseconds(absl::ToUnixMillis(t) - x);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: d = (t - absl::FromUnixMillis(x));
  d = absl::Microseconds(absl::ToUnixMicros(t) - x);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: d = (t - absl::FromUnixMicros(x));
  d = absl::Nanoseconds(absl::ToUnixNanos(t) - x);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: d = (t - absl::FromUnixNanos(x));

  y = x - absl::ToUnixHours(t);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: y = absl::ToInt64Hours(absl::FromUnixHours(x) - t);
  y = x - absl::ToUnixMinutes(t);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: y = absl::ToInt64Minutes(absl::FromUnixMinutes(x) - t);
  y = x - absl::ToUnixSeconds(t);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: y = absl::ToInt64Seconds(absl::FromUnixSeconds(x) - t);
  y = x - absl::ToUnixMillis(t);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: y = absl::ToInt64Milliseconds(absl::FromUnixMillis(x) - t);
  y = x - absl::ToUnixMicros(t);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: y = absl::ToInt64Microseconds(absl::FromUnixMicros(x) - t);
  y = x - absl::ToUnixNanos(t);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: y = absl::ToInt64Nanoseconds(absl::FromUnixNanos(x) - t);

  // Check parenthesis placement
  d = 5 * absl::Seconds(absl::ToUnixSeconds(t) - x);
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: d = 5 * (t - absl::FromUnixSeconds(x));
  d = absl::Seconds(absl::ToUnixSeconds(t) - x) / 5;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: d = (t - absl::FromUnixSeconds(x)) / 5;

  // No extra parens around arguments
  g(absl::Seconds(absl::ToUnixSeconds(t) - x));
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: g(t - absl::FromUnixSeconds(x));
  g(absl::Seconds(x - absl::ToUnixSeconds(t)));
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: g(absl::FromUnixSeconds(x) - t);

  // More complex subexpressions
  d = absl::Hours(absl::ToUnixHours(t) - 5 * x);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: d = (t - absl::FromUnixHours(5 * x));

  // These should not trigger; they are likely bugs
  d = absl::Milliseconds(absl::ToUnixSeconds(t) - x);
  d = absl::Seconds(absl::ToUnixMicros(t) - x);

  // Various macro scenarios
#define SUB(z, t1) z - absl::ToUnixSeconds(t1)
  y = SUB(x, t);
#undef SUB

#define MILLIS(t1) absl::ToUnixMillis(t1)
  y = x - MILLIS(t);
#undef MILLIS

#define HOURS(z) absl::Hours(z)
  d = HOURS(absl::ToUnixHours(t) - x);
#undef HOURS

  // This should match the expression inside the macro invocation.
#define SECONDS(z) absl::Seconds(z)
  d = SECONDS(x - absl::ToUnixSeconds(t));
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: d = SECONDS(absl::ToInt64Seconds(absl::FromUnixSeconds(x) - t));
#undef SECONDS
}

template<typename T>
void func(absl::Time t, T x) {
  absl::Duration d = absl::Seconds(absl::ToUnixSeconds(t) - x);
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: absl::Duration d = t - absl::FromUnixSeconds(x);
}

void g() {
  func(absl::Now(), 5);
}

absl::Duration parens_in_return() {
  absl::Time t;
  int x;

  return absl::Seconds(absl::ToUnixSeconds(t) - x);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: return t - absl::FromUnixSeconds(x);
  return absl::Seconds(x - absl::ToUnixSeconds(t));
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: perform subtraction in the time domain [abseil-time-subtraction]
  // CHECK-FIXES: return absl::FromUnixSeconds(x) - t;
}
