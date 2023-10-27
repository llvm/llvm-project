// RUN: %check_clang_tidy %s misc-unused-parameters %t -- \
// RUN:   -config="{CheckOptions: {misc-unused-parameters.IgnoreVirtual: true}}" --

struct Base {
  int f(int foo) {
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: parameter 'foo' is unused [misc-unused-parameters]
  // CHECK-FIXES: int f(int  /*foo*/) {
    return 5;
  }

  virtual int f2(int foo) {
    return 5;
  }
};

struct Derived : Base {
  int f2(int foo) override {
    return 5;
  }
};
