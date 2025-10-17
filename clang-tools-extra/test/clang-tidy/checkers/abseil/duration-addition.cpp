// RUN: %check_clang_tidy %s abseil-duration-addition %t -- -- -I%S/Inputs

#include "absl/time/time.h"

void f() {
  absl::Time t;
  int i;

  i = absl::ToUnixHours(t) + 5;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: i = absl::ToUnixHours(t + absl::Hours(5));
  i = absl::ToUnixMinutes(t) + 5;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: i = absl::ToUnixMinutes(t + absl::Minutes(5));
  i = absl::ToUnixSeconds(t) + 5;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: i = absl::ToUnixSeconds(t + absl::Seconds(5));
  i = absl::ToUnixMillis(t) + 5;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: i = absl::ToUnixMillis(t + absl::Milliseconds(5));
  i = absl::ToUnixMicros(t) + 5;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: i = absl::ToUnixMicros(t + absl::Microseconds(5));
  i = absl::ToUnixNanos(t) + 5;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: i = absl::ToUnixNanos(t + absl::Nanoseconds(5));

  i = 3 + absl::ToUnixHours(t);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: i = absl::ToUnixHours(absl::Hours(3) + t);
  i = 3 + absl::ToUnixMinutes(t);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: i = absl::ToUnixMinutes(absl::Minutes(3) + t);
  i = 3 + absl::ToUnixSeconds(t);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: i = absl::ToUnixSeconds(absl::Seconds(3) + t);
  i = 3 + absl::ToUnixMillis(t);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: i = absl::ToUnixMillis(absl::Milliseconds(3) + t);
  i = 3 + absl::ToUnixMicros(t);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: i = absl::ToUnixMicros(absl::Microseconds(3) + t);
  i = 3 + absl::ToUnixNanos(t);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: i = absl::ToUnixNanos(absl::Nanoseconds(3) + t);

  // Undoing inverse conversions
  i = absl::ToUnixMicros(t) + absl::ToInt64Microseconds(absl::Seconds(1));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: i = absl::ToUnixMicros(t + absl::Seconds(1));

  // Parens
  i = 3 + (absl::ToUnixHours(t));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: i = absl::ToUnixHours(absl::Hours(3) + t);

  // Float folding
  i = absl::ToUnixSeconds(t) + 5.0;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: i = absl::ToUnixSeconds(t + absl::Seconds(5));

  // We can rewrite the argument of the duration conversion
#define THIRTY absl::FromUnixSeconds(30)
  i = absl::ToUnixSeconds(THIRTY) + 1;
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: i = absl::ToUnixSeconds(THIRTY + absl::Seconds(1));
#undef THIRTY

  // Some other contexts
  if (absl::ToUnixSeconds(t) + 1.0 > 10) {}
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: if (absl::ToUnixSeconds(t + absl::Seconds(1)) > 10) {}

  // These should not match
  i = 5 + 6;
  i = absl::ToUnixSeconds(t) - 1.0;
  i = absl::ToUnixSeconds(t) * 1.0;
  i = absl::ToUnixSeconds(t) / 1.0;
  i += absl::ToInt64Microseconds(absl::Seconds(1));

#define PLUS_FIVE(z) absl::ToUnixSeconds(z) + 5
  i = PLUS_FIVE(t);
#undef PLUS_FIVE
}

// Within a templated function
template<typename T>
void foo(absl::Time t) {
  int i = absl::ToUnixNanos(t) + T{};
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: perform addition in the duration domain [abseil-duration-addition]
  // CHECK-FIXES: int i = absl::ToUnixNanos(t + absl::Nanoseconds(T{}));
}

void g() {
  absl::Time t;
  foo<int>(t);
  foo<double>(t);
}
