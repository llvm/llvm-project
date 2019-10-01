// RUN: %check_clang_tidy %s bugprone-unhandled-self-assignment %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             [{key: bugprone-unhandled-self-assignment.WarnOnlyIfThisHasSuspiciousField, \
// RUN:               value: 0}]}"

// Classes with pointer field are still caught.
class PtrField {
public:
  PtrField &operator=(const PtrField &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:13: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    return *this;
  }

private:
  int *p;
};

// With the option, check catches classes with trivial fields.
class TrivialFields {
public:
  TrivialFields &operator=(const TrivialFields &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:18: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    return *this;
  }

private:
  int m;
  float f;
  double d;
  bool b;
};

// The check warns also when there is no field at all.
// In this case, user-defined copy assignment operator is useless anyway.
class ClassWithoutFields {
public:
  ClassWithoutFields &operator=(const ClassWithoutFields &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:23: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    return *this;
  }
};
