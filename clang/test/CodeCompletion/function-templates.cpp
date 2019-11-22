namespace std {
  template<typename RandomAccessIterator>
  void sort(RandomAccessIterator first, RandomAccessIterator last);

  template<class X, class Y>
  X* dyn_cast(Y *Val);
}

class Foo {
public:
  template<typename T> T &getAs();
};

template <typename T, typename U, typename V>
V doSomething(T t, const U &u, V *v) { return V(); }

void f() {
  std::sort(1, 2);
  Foo().getAs<int>();
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:18:8 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: dyn_cast<<#class X#>>(<#Y *Val#>)
  // CHECK-CC1: sort(<#RandomAccessIterator first#>, <#RandomAccessIterator last#>
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:19:9 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
  // CHECK-CC2: getAs<<#typename T#>>()
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:19:22 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
  // CHECK-CC3: [#V#]doSomething(<#T t#>, <#const U &u#>, <#V *v#>)
}
