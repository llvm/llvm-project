// RUN: %check_clang_tidy -check-suffix=REMOVE %s modernize-use-auto %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-auto.RemoveStars: 'true', modernize-use-auto.MinTypeNameLength: '0'}}"
// RUN: %check_clang_tidy %s modernize-use-auto %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-auto.RemoveStars: 'false', modernize-use-auto.MinTypeNameLength: '0'}}"

void pointerToFunction() {
  void (*(*(f1)))() = static_cast<void (**)()>(nullptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing
  // CHECK-FIXES-REMOVE: auto f1 = static_cast<void (**)()>(nullptr);
  // CHECK-FIXES: auto *f1 = static_cast<void (**)()>(nullptr);
}

void pointerToArray() {
  int(*a1)[2] = new int[10][2];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing
  // CHECK-FIXES-REMOVE: auto a1 = new int[10][2];
  // CHECK-FIXES: auto *a1 = new int[10][2];
}

void memberFunctionPointer() {
  class A {
    void f();
  };
  void(A::* a1)() = static_cast<void(A::*)()>(nullptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when initializing
  // CHECK-FIXES-REMOVE: auto a1 = static_cast<void(A::*)()>(nullptr);
  // CHECK-FIXES: auto *a1 = static_cast<void(A::*)()>(nullptr);
}

