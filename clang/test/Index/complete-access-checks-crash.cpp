struct Base {
protected:
  bool bar();
};
struct Derived : Base {
};

struct X {
  int foo() {
    Derived(). // RUN: c-index-test -code-completion-at=%s:10:15 %s | FileCheck %s
    // CHECK: bar{{.*}}(inaccessible)
  }
};
