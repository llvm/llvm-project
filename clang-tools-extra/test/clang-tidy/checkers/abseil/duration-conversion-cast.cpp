// RUN: %check_clang_tidy %s abseil-duration-conversion-cast %t -- -- -I%S/Inputs

#include "absl/time/time.h"

void f() {
  absl::Duration d1;
  double x;
  int i;

  i = static_cast<int>(absl::ToDoubleHours(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: duration should be converted directly to an integer rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToInt64Hours(d1);
  x = static_cast<float>(absl::ToInt64Hours(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: duration should be converted directly to a floating-point number rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToDoubleHours(d1);
  i = static_cast<int>(absl::ToDoubleMinutes(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: duration should be converted directly to an integer rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToInt64Minutes(d1);
  x = static_cast<float>(absl::ToInt64Minutes(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: duration should be converted directly to a floating-point number rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToDoubleMinutes(d1);
  i = static_cast<int>(absl::ToDoubleSeconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: duration should be converted directly to an integer rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToInt64Seconds(d1);
  x = static_cast<float>(absl::ToInt64Seconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: duration should be converted directly to a floating-point number rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToDoubleSeconds(d1);
  i = static_cast<int>(absl::ToDoubleMilliseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: duration should be converted directly to an integer rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToInt64Milliseconds(d1);
  x = static_cast<float>(absl::ToInt64Milliseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: duration should be converted directly to a floating-point number rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToDoubleMilliseconds(d1);
  i = static_cast<int>(absl::ToDoubleMicroseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: duration should be converted directly to an integer rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToInt64Microseconds(d1);
  x = static_cast<float>(absl::ToInt64Microseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: duration should be converted directly to a floating-point number rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToDoubleMicroseconds(d1);
  i = static_cast<int>(absl::ToDoubleNanoseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: duration should be converted directly to an integer rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToInt64Nanoseconds(d1);
  x = static_cast<float>(absl::ToInt64Nanoseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: duration should be converted directly to a floating-point number rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToDoubleNanoseconds(d1);

  // Functional-style casts
  i = int(absl::ToDoubleHours(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: duration should be converted directly to an integer rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToInt64Hours(d1);
  x = float(absl::ToInt64Microseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: duration should be converted directly to a floating-point number rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToDoubleMicroseconds(d1);

  // C-style casts
  i = (int) absl::ToDoubleHours(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: duration should be converted directly to an integer rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToInt64Hours(d1);
  x = (float) absl::ToInt64Microseconds(d1);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: duration should be converted directly to a floating-point number rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToDoubleMicroseconds(d1);

  // Type aliasing
  typedef int FancyInt;
  typedef float FancyFloat;

  FancyInt j = static_cast<FancyInt>(absl::ToDoubleHours(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: duration should be converted directly to an integer rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToInt64Hours(d1);
  FancyFloat k = static_cast<FancyFloat>(absl::ToInt64Microseconds(d1));
  // CHECK-MESSAGES: [[@LINE-1]]:18: warning: duration should be converted directly to a floating-point number rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: absl::ToDoubleMicroseconds(d1);

  // Macro handling
  // We want to transform things in macro arguments
#define EXTERNAL(x) (x) + 5
  i = EXTERNAL(static_cast<int>(absl::ToDoubleSeconds(d1)));
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: duration should be converted directly to an integer rather than through a type cast [abseil-duration-conversion-cast]
  // CHECK-FIXES: EXTERNAL(absl::ToInt64Seconds(d1));
#undef EXTERNAL

  // We don't want to transform this which get split across macro boundaries
#define SPLIT(x) static_cast<int>((x)) + 5
  i = SPLIT(absl::ToDoubleSeconds(d1));
#undef SPLIT

  // We also don't want to transform things inside of a macro definition
#define INTERNAL(x) static_cast<int>(absl::ToDoubleSeconds((x))) + 5
  i = INTERNAL(d1);
#undef INTERNAL

  // These shouldn't be converted
  i = static_cast<int>(absl::ToInt64Seconds(d1));
  i = static_cast<float>(absl::ToDoubleSeconds(d1));
}
