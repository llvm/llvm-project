// RUN: %check_clang_tidy %s abseil-duration-comparison %t -- -- -I%S/Inputs

#include "absl/time/time.h"

void f() {
  double x;
  absl::Duration d1, d2;
  bool b;
  absl::Time t1, t2;

  // Check against the RHS
  b = x > absl::ToDoubleSeconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Seconds(x) > d1;
  b = x >= absl::ToDoubleSeconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Seconds(x) >= d1;
  b = x == absl::ToDoubleSeconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Seconds(x) == d1;
  b = x <= absl::ToDoubleSeconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Seconds(x) <= d1;
  b = x < absl::ToDoubleSeconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Seconds(x) < d1;
  b = x == absl::ToDoubleSeconds(t1 - t2);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Seconds(x) == t1 - t2;
  b = absl::ToDoubleSeconds(d1) > absl::ToDoubleSeconds(d2);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: d1 > d2;

  // Check against the LHS
  b = absl::ToDoubleSeconds(d1) < x;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: d1 < absl::Seconds(x);
  b = absl::ToDoubleSeconds(d1) <= x;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: d1 <= absl::Seconds(x);
  b = absl::ToDoubleSeconds(d1) == x;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: d1 == absl::Seconds(x);
  b = absl::ToDoubleSeconds(d1) >= x;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: d1 >= absl::Seconds(x);
  b = absl::ToDoubleSeconds(d1) > x;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: d1 > absl::Seconds(x);

  // Comparison against zero
  b = absl::ToDoubleSeconds(d1) < 0.0;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: d1 < absl::ZeroDuration();
  b = absl::ToDoubleSeconds(d1) < 0;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: d1 < absl::ZeroDuration();

  // Scales other than Seconds
  b = x > absl::ToDoubleMicroseconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Microseconds(x) > d1;
  b = x >= absl::ToDoubleMilliseconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Milliseconds(x) >= d1;
  b = x == absl::ToDoubleNanoseconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Nanoseconds(x) == d1;
  b = x <= absl::ToDoubleMinutes(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Minutes(x) <= d1;
  b = x < absl::ToDoubleHours(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Hours(x) < d1;

  // Integer comparisons
  b = x > absl::ToInt64Microseconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Microseconds(x) > d1;
  b = x >= absl::ToInt64Milliseconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Milliseconds(x) >= d1;
  b = x == absl::ToInt64Nanoseconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Nanoseconds(x) == d1;
  b = x == absl::ToInt64Seconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Seconds(x) == d1;
  b = x <= absl::ToInt64Minutes(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Minutes(x) <= d1;
  b = x < absl::ToInt64Hours(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Hours(x) < d1;

  // Other abseil-duration checks folded into this one
  b = static_cast<double>(5) > absl::ToDoubleSeconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Seconds(5) > d1;
  b = double(5) > absl::ToDoubleSeconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Seconds(5) > d1;
  b = float(5) > absl::ToDoubleSeconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Seconds(5) > d1;
  b = ((double)5) > absl::ToDoubleSeconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Seconds(5) > d1;
  b = 5.0 > absl::ToDoubleSeconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Seconds(5) > d1;

  // A long expression
  bool some_condition;
  int very_very_very_very_long_variable_name;
  absl::Duration SomeDuration;
  if (some_condition && very_very_very_very_long_variable_name
     < absl::ToDoubleSeconds(SomeDuration)) {
  // CHECK-MESSAGES: [[@LINE-2]]:25: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: if (some_condition && absl::Seconds(very_very_very_very_long_variable_name) < SomeDuration) {
    return;
  }

  // A complex expression
  int y;
  b = (y + 5) * 10 > absl::ToDoubleMilliseconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform comparison in the duration domain [abseil-duration-comparison]
  // CHECK-FIXES: absl::Milliseconds((y + 5) * 10) > d1;

  // These should not match
  b = 6 < 4;

#define TODOUBLE(x) absl::ToDoubleSeconds(x)
  b = 5.0 > TODOUBLE(d1);
#undef TODOUBLE
#define THIRTY 30.0
  b = THIRTY > absl::ToDoubleSeconds(d1);
#undef THIRTY
}
