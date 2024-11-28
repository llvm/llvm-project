// RUN: %check_clang_tidy %s modernize-cleanup-static-cast %t

void foo(unsigned long x) {}
void bar(int x) {}

void test() {
  unsigned long s = 42;
  foo(static_cast<unsigned long>(s));  // Should warn
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: redundant static_cast to the same type 'unsigned long' [modernize-cleanup-static-cast]
  // CHECK-FIXES: foo(s);

  // Different types - no warning
  int i = 42;
  foo(static_cast<unsigned long>(i));

  // Test with typedef - should warn
  typedef unsigned long my_ul_t;
  my_ul_t ms = 42;
  foo(static_cast<unsigned long>(ms));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: redundant static_cast to the same type 'unsigned long' [modernize-cleanup-static-cast]
  // CHECK-FIXES: foo(ms);
}

// Template - no warnings
template<typename T>
void template_function(T value) {
  foo(static_cast<unsigned long>(value));
}

void test_templates() {
  template_function<unsigned long>(42);
  template_function<int>(42);
}

// Test multiple casts
void test_multiple() {
  unsigned long s = 42;
  foo(static_cast<unsigned long>(static_cast<unsigned long>(s)));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: redundant static_cast to the same type 'unsigned long' [modernize-cleanup-static-cast]
  // CHECK-MESSAGES: [[@LINE-2]]:34: warning: redundant static_cast to the same type 'unsigned long' [modernize-cleanup-static-cast]
  // CHECK-FIXES: foo(static_cast<unsigned long>(s));
}