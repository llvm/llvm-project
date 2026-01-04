// RUN: %check_clang_tidy -std=c++98,c++03 %s readability-redundant-typename %t \
// RUN:   -- -- -fno-delayed-template-parsing

struct NotDependent {
  typedef int R;
  template <typename = int>
  struct T {};
};

template <typename T>
typename T::R f() {
  static_cast<typename T::R>(0);

  typename NotDependent::R NotDependentVar;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES: NotDependent::R NotDependentVar;

  typename NotDependent::T<int> V1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES: NotDependent::T<int> V1;

  void notDependentFunctionDeclaration(typename NotDependent::R);
  // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES: void notDependentFunctionDeclaration(NotDependent::R);
}
