// RUN: %check_clang_tidy -std=c++11-or-later %s performance-function-pointer-equality %t

void f();
void g();

void test_equality_static() {
  bool b1 = (f == g);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: function pointer equality comparison is unsound with '--icf=all' enabled [performance-function-pointer-equality]
  bool b2 = (f != g);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: function pointer equality comparison is unsound with '--icf=all' enabled [performance-function-pointer-equality]
}

void test_equality(void (*fp1)(), void (*fp2)()) {
  bool b3 = (fp1 == fp2);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: function pointer equality comparison is unsound with '--icf=all' enabled [performance-function-pointer-equality]
  bool b4 = (fp1 != fp2);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: function pointer equality comparison is unsound with '--icf=all' enabled [performance-function-pointer-equality]
  bool b5 = (fp1 == f);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: function pointer equality comparison is unsound with '--icf=all' enabled [performance-function-pointer-equality]

  void (&rf)() = f;
  void (&rg)() = g;
  bool b6 = (rf == rg);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: function pointer equality comparison is unsound with '--icf=all' enabled [performance-function-pointer-equality]

  // Null pointer comparisons are allowed.
  bool n1 = (f == nullptr);
  bool n2 = (f != nullptr);
  bool n3 = (nullptr == fp1);
  bool n4 = (fp1 != 0);
  bool n5 = (fp1 == static_cast<void (*)()>(nullptr));
}
