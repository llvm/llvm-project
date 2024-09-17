// RUN: %check_clang_tidy %s cert-flp30-c %t

float g(void);
int c(float);
float f = 1.0f;

void match(void) {

  for (float x = 0.1f; x <= 1.0f; x += 0.1f) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: loop induction expression should not have floating-point type [cert-flp30-c]

  for (; f > 0; --f) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: loop induction expression should not have floating-point type [cert-flp30-c]

  for (float x = 0.0f; c(x); x = g()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: loop induction expression should not have floating-point type [cert-flp30-c]

  for (int i=0; i < 10 && f < 2.0f; f++, i++) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: loop induction expression should not have floating-point type [cert-flp30-c]
  // CHECK-MESSAGES: :5:1: note: floating-point type loop induction variable
}

void not_match(void) {
  for (int i = 0; i < 10; i += 1.0f) {}
  for (int i = 0; i < 10; ++i) {}
  for (int i = 0; i < 10; ++i, f++) {}
  for (int i = 0; f < 10.f; ++i) {}
}
