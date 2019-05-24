// RUN: %check_clang_tidy %s cert-oop54-cpp %t

// Test whether bugprone-unhandled-self-assignment.WarnOnlyIfThisHasSuspiciousField option is set correctly.
class TrivialFields {
public:
  TrivialFields &operator=(const TrivialFields &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:18: warning: operator=() does not handle self-assignment properly [cert-oop54-cpp]
    return *this;
  }

private:
  int m;
  float f;
  double d;
  bool b;
};
