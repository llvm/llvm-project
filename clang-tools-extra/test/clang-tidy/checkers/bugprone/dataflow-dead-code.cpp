// RUN: %check_clang_tidy %s bugprone-dataflow-dead-code %t

void simple_cases(bool a) {
  if (a) {
    if (a) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: dead code - branching condition is always true
    while (!a) {}
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: dead code - branching condition is always false
    // fixme: false positive
    // CHECK-MESSAGES: :[[@LINE-3]]:13: warning: dead code - branching condition is always true
  }
}

void transitive(bool a, bool b) {
  if (a) {
    return;
  } else if (a == b) {
    // fixme: false positive
    // CHECK-MESSAGES: :[[@LINE-2]]:14: warning: dead code - branching condition is always false
    if (b) { 
      // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: dead code - branching condition is always false
      return;
    }
  }
}

#define assert(x) do {} while(false)

void assert_excluded(int *x) {
  assert(x); // no-warning
}

extern bool random_bool();

void empty_forloop_excluded() {
  for (;;) { // no-warning
    if (random_bool()) { // no-warning
      return;
    }
  }
}

struct S {
  bool a;
  void change_a() { a = random_bool(); }
};

void false_positive_classes(S s) {
  if (s.a) {
    return;
  }
  s.change_a();
  if (s.a) {} // fixme: false positive
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: dead code - branching condition is always false
}