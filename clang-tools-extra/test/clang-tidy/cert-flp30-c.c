// RUN: %check_clang_tidy %s cert-flp30-c %t

float g(void);

void func(void) {
  for (float x = 0.1f; x <= 1.0f; x += 0.1f) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: loop induction expression should not have floating-point type [cert-flp30-c]

  float f = 1.0f;
  for (; f > 0; --f) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: loop induction expression

  for (;;g()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: loop induction expression

  for (int i = 0; i < 10; i += 1.0f) {}

  for (int i = 0; i < 10; ++i) {}
}
