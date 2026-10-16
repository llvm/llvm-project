// RUN: %check_clang_tidy %s abseil-duration-factory-scale %t -- -- -I%S/Inputs

#include "absl/time/time.h"

namespace std { typedef long long int64_t; }
using int64_t = std::int64_t;

void ScaleTest() {
  absl::Duration d;

  // Zeroes
  d = absl::Hours(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use ZeroDuration() for zero-length time intervals [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::ZeroDuration();
  d = absl::Minutes(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use ZeroDuration() for zero-length time intervals [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::ZeroDuration();
  d = absl::Seconds(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use ZeroDuration() for zero-length time intervals [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::ZeroDuration();
  d = absl::Milliseconds(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use ZeroDuration() for zero-length time intervals [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::ZeroDuration();
  d = absl::Microseconds(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use ZeroDuration() for zero-length time intervals [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::ZeroDuration();
  d = absl::Nanoseconds(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use ZeroDuration() for zero-length time intervals [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::ZeroDuration();
  d = absl::Seconds(0.0);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use ZeroDuration() for zero-length time intervals [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::ZeroDuration();
  d = absl::Seconds(0x0.000001p-126f);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use ZeroDuration() for zero-length time intervals [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::ZeroDuration();
  d = absl::Seconds(int{0});
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use ZeroDuration() for zero-length time intervals [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::ZeroDuration();
  d = absl::Seconds(int64_t{0});
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use ZeroDuration() for zero-length time intervals [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::ZeroDuration();
  d = absl::Seconds(float{0});
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use ZeroDuration() for zero-length time intervals [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::ZeroDuration();

  // Fold seconds into minutes
  d = absl::Seconds(30 * 60);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: internal duration scaling can be removed [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::Minutes(30);
  d = absl::Seconds(60 * 30);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: internal duration scaling can be removed [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::Minutes(30);

  // Try a few more exotic multiplications
  d = absl::Seconds(60 * 30 * 60);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: internal duration scaling can be removed [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::Minutes(60 * 30);
  d = absl::Seconds(1e-3 * 30);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: internal duration scaling can be removed [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::Milliseconds(30);
  d = absl::Milliseconds(30 * 1000);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: internal duration scaling can be removed [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::Seconds(30);
  d = absl::Milliseconds(30 * 0.001);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: internal duration scaling can be removed [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::Microseconds(30);

  // Multiple steps
  d = absl::Seconds(5 * 3600);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: internal duration scaling can be removed [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::Hours(5);
  d = absl::Microseconds(5 * 1e6);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: internal duration scaling can be removed [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::Seconds(5);
  d = absl::Seconds(5 * 1e-6);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: internal duration scaling can be removed [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::Microseconds(5);
  d = absl::Microseconds(5 * 1000000);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: internal duration scaling can be removed [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::Seconds(5);

  // Division
  d = absl::Hours(30 / 60.);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: internal duration scaling can be removed [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::Minutes(30);
  d = absl::Seconds(30 / 1000.);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: internal duration scaling can be removed [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::Milliseconds(30);
  d = absl::Milliseconds(30 / 1e3);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: internal duration scaling can be removed [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::Microseconds(30);
  d = absl::Seconds(30 / 1e6);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: internal duration scaling can be removed [abseil-duration-factory-scale]
  // CHECK-FIXES: d = absl::Microseconds(30);

  // None of these should trigger the check
  d = absl::Seconds(60);
  d = absl::Seconds(int{60});
  d = absl::Seconds(float{60});
  d = absl::Seconds(60 + 30);
  d = absl::Seconds(60 - 30);
  d = absl::Seconds(50 * 30);
  d = absl::Hours(60 * 60);
  d = absl::Nanoseconds(1e-3 * 30);
  d = absl::Seconds(1000 / 30);
  // We don't support division by integers, which could cause rounding
  d = absl::Seconds(10 / 1000);
  d = absl::Seconds(30 / 50);

#define EXPRESSION 30 * 60
  d = absl::Seconds(EXPRESSION);
#undef EXPRESSION

// This should not trigger
#define HOURS(x) absl::Minutes(60 * x)
  d = HOURS(40);
#undef HOURS
}
