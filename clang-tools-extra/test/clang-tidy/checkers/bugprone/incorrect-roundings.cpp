// RUN: %check_clang_tidy %s bugprone-incorrect-roundings %t

void b(int x) {}

void f1() {
  float f;
  double d;
  long double ld;
  int x;

  x = (d + 0.5);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: casting (double + 0.5) to integer leads to incorrect rounding; consider using lround (#include <cmath>) instead [bugprone-incorrect-roundings]
  x = (d + 0.5f);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: casting (double + 0.5)
  x = (d + 0.5L);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: casting (double + 0.5)
  x = (f + 0.5);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: casting (double + 0.5)
  x = (f + 0.5f);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: casting (double + 0.5)
  x = (f + 0.5L);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: casting (double + 0.5)
  x = (0.5 + d);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: casting (double + 0.5)
  x = (0.5f + d);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: casting (double + 0.5)
  x = (0.5L + d);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: casting (double + 0.5)
  x = (0.5 + ld);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: casting (double + 0.5)
  x = (0.5f + ld);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: casting (double + 0.5)
  x = (0.5L + ld);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: casting (double + 0.5)
  x = (0.5 + f);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: casting (double + 0.5)
  x = (0.5f + f);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: casting (double + 0.5)
  x = (0.5L + f);
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: casting (double + 0.5)
  x = (int)(d + 0.5);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(d + 0.5f);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(d + 0.5L);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(ld + 0.5);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(ld + 0.5f);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(ld + 0.5L);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(f + 0.5);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(f + 0.5f);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(f + 0.5L);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(0.5 + d);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(0.5f + d);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(0.5L + d);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(0.5 + ld);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(0.5f + ld);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(0.5L + ld);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(0.5 + f);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(0.5f + f);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = (int)(0.5L + f);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: casting (double + 0.5)
  x = static_cast<int>(d + 0.5);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(d + 0.5f);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(d + 0.5L);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(ld + 0.5);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(ld + 0.5f);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(ld + 0.5L);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(f + 0.5);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(f + 0.5f);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(f + 0.5L);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(0.5 + d);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(0.5f + d);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(0.5L + d);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(0.5 + ld);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(0.5f + ld);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(0.5L + ld);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(0.5 + f);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(0.5f + f);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)
  x = static_cast<int>(0.5L + f);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: casting (double + 0.5)

  // Don't warn if constant is not 0.5.
  x = (int)(d + 0.4999);
  x = (int)(d + 0.4999f);
  x = (int)(d + 0.4999L);
  x = (int)(0.4999 + d);
  x = (int)(0.4999f + d);
  x = (int)(0.4999L + d);

  x = (int)(d + -0.5);
  x = (int)(-0.5 + d);
  x = (int)(d + -0.5f);
  x = (int)(-0.5f + d);
  x = (int)(d + -0.5L);
  x = (int)(-0.5L + d);

  x = (int)(d + 0.5001);
  x = (int)(d + 0.5001f);
  x = (int)(d + 0.5001L);
  x = (int)(0.5001 + d);
  x = (int)(0.5001f + d);
  x = (int)(0.5001L + d);

  // Don't warn if binary operator is not directly beneath cast.
  x = (int)(1 + (0.5 + f));
}
