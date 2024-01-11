// RUN: %check_clang_tidy %s readability-ConditionalToStdMinMax %t

void foo() {
  int value1,value2;

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: use std::max instead of < [readability-ConditionalToStdMinMax]
  if (value1 < value2)
    value1 = value2; // CHECK-FIXES: value1 = std::max(value1, value2);

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: use std::min instead of < [readability-ConditionalToStdMinMax]
  if (value1 < value2)
    value2 = value1; // CHECK-FIXES: value2 = std::min(value1, value2);

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: use std::min instead of > [readability-ConditionalToStdMinMax]
  if (value2 > value1)
    value2 = value1; // CHECK-FIXES: value2 = std::min(value2, value1);

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: use std::max instead of > [readability-ConditionalToStdMinMax]
  if (value2 > value1)
    value1 = value2; // CHECK-FIXES: value1 = std::max(value2, value1);

  // No suggestion needed here
  if (value1 == value2)
    value1 = value2;

  
}