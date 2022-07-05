// RUN: %check_clang_tidy %s misc-confusable-identifiers %t

int fo;
int 𝐟o;
// CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: '𝐟o' is confusable with 'fo' [misc-confusable-identifiers]
// CHECK-MESSAGES: :[[#@LINE-3]]:5: note: other declaration found here

void no() {
  int 𝐟oo;
}

void worry() {
  int foo;
}
int 𝐟i;
int fi;
// CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: 'fi' is confusable with '𝐟i' [misc-confusable-identifiers]
// CHECK-MESSAGES: :[[#@LINE-3]]:5: note: other declaration found here

bool f0(const char *q1, const char *ql) {
  // CHECK-MESSAGES: :[[#@LINE-1]]:37: warning: 'ql' is confusable with 'q1' [misc-confusable-identifiers]
  // CHECK-MESSAGES: :[[#@LINE-2]]:21: note: other declaration found here
  return q1 < ql;
}

// should not print anything
namespace ns {
struct Foo {};
} // namespace ns
auto f = ns::Foo();

struct Test {
  void f1(const char *pl);
};

bool f2(const char *p1, const char *ql) {
  return p1 < ql;
}

bool f3(const char *q0, const char *q1) {
  return q0 < q1;
}

template <typename i1>
struct S {
  template <typename il>
  void f4() {}
  // CHECK-MESSAGES: :[[#@LINE-2]]:22: warning: 'il' is confusable with 'i1' [misc-confusable-identifiers]
  // CHECK-MESSAGES: :[[#@LINE-5]]:20: note: other declaration found here
};

template <typename i1>
void f5(int il) {
  // CHECK-MESSAGES: :[[#@LINE-1]]:13: warning: 'il' is confusable with 'i1' [misc-confusable-identifiers]
  // CHECK-MESSAGES: :[[#@LINE-3]]:20: note: other declaration found here
}

template <typename O0>
void f6() {
  int OO = 0;
  // CHECK-MESSAGES: :[[#@LINE-1]]:7: warning: 'OO' is confusable with 'O0' [misc-confusable-identifiers]
  // CHECK-MESSAGES: :[[#@LINE-4]]:20: note: other declaration found here
}
int OO = 0; // no warning, not same scope as f6

namespace f7 {
int i1;
}

namespace f8 {
int il; // no warning, different namespace
}

namespace f7 {
int il;
// CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: 'il' is confusable with 'i1' [misc-confusable-identifiers]
// CHECK-MESSAGES: :[[#@LINE-10]]:5: note: other declaration found here
} // namespace f7

template <typename t1, typename tl>
// CHECK-MESSAGES: :[[#@LINE-1]]:33: warning: 'tl' is confusable with 't1' [misc-confusable-identifiers]
// CHECK-MESSAGES: :[[#@LINE-2]]:20: note: other declaration found here
void f9();

struct Base0 {
  virtual void mO0();

private:
  void mII();
};

struct Derived0 : Base0 {
  void mOO();
  // CHECK-MESSAGES: :[[#@LINE-1]]:8: warning: 'mOO' is confusable with 'mO0' [misc-confusable-identifiers]
  // CHECK-MESSAGES: :[[#@LINE-9]]:16: note: other declaration found here

  void mI1(); // no warning: mII is private
};

struct Base1 {
  long mO0;

private:
  long mII;
};

struct Derived1 : Base1 {
  long mOO;
  // CHECK-MESSAGES: :[[#@LINE-1]]:8: warning: 'mOO' is confusable with 'mO0' [misc-confusable-identifiers]
  // CHECK-MESSAGES: :[[#@LINE-9]]:8: note: other declaration found here

  long mI1(); // no warning: mII is private
};
