// RUN: %check_clang_tidy -std=c++11-or-later %s misc-function-pointer-equality %t

void f();
void g();

void test_equality_static() {
  bool b1 = (f == g);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: function pointer equality comparison is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  bool b2 = (f != g);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: function pointer equality comparison is unsound with '--icf=all' enabled [misc-function-pointer-equality]
}

void test_equality(void (*fp1)(), void (*fp2)()) {
  bool b3 = (fp1 == fp2);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: function pointer equality comparison is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  bool b4 = (fp1 != fp2);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: function pointer equality comparison is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  bool b5 = (fp1 == f);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: function pointer equality comparison is unsound with '--icf=all' enabled [misc-function-pointer-equality]

  void (&rf)() = f;
  void (&rg)() = g;
  bool b6 = (rf == rg);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: function pointer equality comparison is unsound with '--icf=all' enabled [misc-function-pointer-equality]

  // Null pointer comparisons are allowed.
  bool n1 = (f == nullptr);
  bool n2 = (f != nullptr);
  bool n3 = (nullptr == fp1);
  bool n4 = (fp1 != 0);
  bool n5 = (fp1 == static_cast<void (*)()>(nullptr));
}

class Foo
{
public:
  int MyMethod1() {
    return 1;
  }
  int MyMethod2() {
    return 2;
  }
  static void MyStaticMethod1() {
    return;
  }
  static void MyStaticMethod2() {
    return;
  }
};

void test_member_function_equality() {
  bool a1 = &Foo::MyMethod1 == &Foo::MyMethod2;
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: function pointer equality comparison is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  bool b1 = &Foo::MyMethod1 != &Foo::MyMethod2;
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: function pointer equality comparison is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  bool a2 = &Foo::MyStaticMethod1 == &Foo::MyStaticMethod2;
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: function pointer equality comparison is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  bool b2 = &Foo::MyStaticMethod1 != &Foo::MyStaticMethod2;
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: function pointer equality comparison is unsound with '--icf=all' enabled [misc-function-pointer-equality]
}

bool weird_nullptr_edgecase(decltype(nullptr) a, void (*b)()) {
  // Should not report an issue, as `a` is a nullptr.
  return a == b;
}

void test_nullptr_t() {
  weird_nullptr_edgecase(nullptr, &f);
}

void test_static_assert() {
  // Can never lead to unsoundness due to code folding.
  static_assert(f != (g));
  static_assert((&f) != &g);
  sizeof(f == g);
  decltype(f != g);
}
