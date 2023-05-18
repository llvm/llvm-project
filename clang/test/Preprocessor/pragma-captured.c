// RUN: %clang_cc1 -E %s | FileCheck %s

// Test pragma clang __debug captured, for Captured Statements

void test1()
{
  #pragma clang __debug captured
  {
  }
// CHECK: void test1()
// CHECK: {
// CHECK: #pragma clang __debug captured
}
