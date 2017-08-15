template <class T>
struct unique_ptr {
  typedef T* pointer;

  void reset(pointer ptr = pointer());
};

void test() {
  unique_ptr<int> x;
  x.
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:10:5 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: [#void#]reset({#<#unique_ptr<int>::pointer ptr = pointer()#>#})
}
