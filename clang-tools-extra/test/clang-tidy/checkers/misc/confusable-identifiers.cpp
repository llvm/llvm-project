// RUN: %check_clang_tidy %s misc-confusable-identifiers %t

int l0;
int lO;
// CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: 'lO' is confusable with 'l0' [misc-confusable-identifiers]
// CHECK-MESSAGES: :[[#@LINE-3]]:5: note: other declaration found here

void no() {
  int 𝐟oo;
}

void worry() {
  int foo;
}
int l1;
int ll;
// CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: 'll' is confusable with 'l1' [misc-confusable-identifiers]
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

namespace different_contexts {
  // No warning for names in unrelated contexts.
  template <typename u1> void different_templates_1();
  template <typename ul> void different_templates_2();
  namespace inner {
    int ul;
  }
}

namespace same_template {
  template <typename u1, typename ul> using T = int;
  // CHECK-MESSAGES: :[[#@LINE-1]]:35: warning: 'ul' is confusable with 'u1' [misc-confusable-identifiers]
  // CHECK-MESSAGES: :[[#@LINE-2]]:22: note: other declaration found here

  template <typename v1, typename vl> int n;
  // CHECK-MESSAGES: :[[#@LINE-1]]:35: warning: 'vl' is confusable with 'v1' [misc-confusable-identifiers]
  // CHECK-MESSAGES: :[[#@LINE-2]]:22: note: other declaration found here

  template <typename w1> int wl;
  // CHECK-MESSAGES: :[[#@LINE-1]]:22: warning: 'w1' is confusable with 'wl' [misc-confusable-identifiers]
  // CHECK-MESSAGES: :[[#@LINE-2]]:30: note: other declaration found here

  int xl;
  template <typename x1> int x;
  // CHECK-MESSAGES: :[[#@LINE-1]]:22: warning: 'x1' is confusable with 'xl' [misc-confusable-identifiers]
  // CHECK-MESSAGES: :[[#@LINE-3]]:7: note: other declaration found here
}

namespace f10 {
int il;
namespace inner {
  int i1;
  // CHECK-MESSAGES: :[[#@LINE-1]]:7: warning: 'i1' is confusable with 'il' [misc-confusable-identifiers]
  // CHECK-MESSAGES: :[[#@LINE-4]]:5: note: other declaration found here
  int j1;
  // CHECK-MESSAGES: :[[#@LINE-1]]:7: warning: 'j1' is confusable with 'jl' [misc-confusable-identifiers]
  // CHECK-MESSAGES: :[[#@LINE+2]]:5: note: other declaration found here
}
int jl;
}

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

struct Base2 {
  long nO0;

private:
  long nII;
};

struct Mid2 : Base0, Base1, Base2 {
};

struct Derived2 : Mid2 {
  long nOO;
  // CHECK-MESSAGES: :[[#@LINE-1]]:8: warning: 'nOO' is confusable with 'nO0' [misc-confusable-identifiers]
  // CHECK-MESSAGES: :[[#@LINE-12]]:8: note: other declaration found here

  long nI1(); // no warning: mII is private
};
