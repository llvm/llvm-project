// RUN: %check_clang_tidy %s abseil-duration-subtraction %t -- -- -I %S/Inputs

#include "absl/time/time.h"

void f() {
  double x;
  absl::Duration d, d1, d2;

  x = absl::ToDoubleSeconds(d) - 1.0;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the duration domain [abseil-duration-subtraction]
  // CHECK-FIXES: x = absl::ToDoubleSeconds(d - absl::Seconds(1));
  x = absl::ToDoubleSeconds(d) - absl::ToDoubleSeconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the duration domain [abseil-duration-subtraction]
  // CHECK-FIXES: x = absl::ToDoubleSeconds(d - d1);
  x = absl::ToDoubleSeconds(d) - 6.5 - 8.0;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the duration domain [abseil-duration-subtraction]
  // CHECK-FIXES: x = absl::ToDoubleSeconds(d - absl::Seconds(6.5)) - 8.0;
  x = absl::ToDoubleHours(d) - 1.0;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the duration domain [abseil-duration-subtraction]
  // CHECK-FIXES: x = absl::ToDoubleHours(d - absl::Hours(1));
  x = absl::ToDoubleMinutes(d) - 1;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the duration domain [abseil-duration-subtraction]
  // CHECK-FIXES: x = absl::ToDoubleMinutes(d - absl::Minutes(1));
  x = absl::ToDoubleMilliseconds(d) - 9;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the duration domain [abseil-duration-subtraction]
  // CHECK-FIXES: x = absl::ToDoubleMilliseconds(d - absl::Milliseconds(9));
  x = absl::ToDoubleMicroseconds(d) - 9;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the duration domain [abseil-duration-subtraction]
  // CHECK-FIXES: x = absl::ToDoubleMicroseconds(d - absl::Microseconds(9));
  x = absl::ToDoubleNanoseconds(d) - 42;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the duration domain [abseil-duration-subtraction]
  // CHECK-FIXES: x = absl::ToDoubleNanoseconds(d - absl::Nanoseconds(42));

  // We can rewrite the argument of the duration conversion
#define THIRTY absl::Seconds(30)
  x = absl::ToDoubleSeconds(THIRTY) - 1.0;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the duration domain [abseil-duration-subtraction]
  // CHECK-FIXES: x = absl::ToDoubleSeconds(THIRTY - absl::Seconds(1));
#undef THIRTY

  // Some other contexts
  if (absl::ToDoubleSeconds(d) - 1.0 > 10) {}
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the duration domain [abseil-duration-subtraction]
  // CHECK-FIXES: if (absl::ToDoubleSeconds(d - absl::Seconds(1)) > 10) {}

  // A nested occurrence
  x = absl::ToDoubleSeconds(d) - absl::ToDoubleSeconds(absl::Seconds(5));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the duration domain [abseil-duration-subtraction]
  // CHECK-FIXES: x = absl::ToDoubleSeconds(d - absl::Seconds(5));
  x = absl::ToDoubleSeconds(d) - absl::ToDoubleSeconds(absl::Seconds(absl::ToDoubleSeconds(d1)));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform subtraction in the duration domain [abseil-duration-subtraction]
  // CHECK-FIXES: x = absl::ToDoubleSeconds(d - absl::Seconds(absl::ToDoubleSeconds(d1)));

  // These should not match
  x = 5 - 6;
  x = 4 - absl::ToDoubleSeconds(d) - 6.5 - 8.0;
  x = absl::ToDoubleSeconds(d) + 1.0;
  x = absl::ToDoubleSeconds(d) * 1.0;
  x = absl::ToDoubleSeconds(d) / 1.0;

#define MINUS_FIVE(z) absl::ToDoubleSeconds(z) - 5
  x = MINUS_FIVE(d);
#undef MINUS_FIVE
}
