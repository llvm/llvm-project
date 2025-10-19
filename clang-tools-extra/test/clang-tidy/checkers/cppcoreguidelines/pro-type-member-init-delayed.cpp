// RUN: %check_clang_tidy %s cppcoreguidelines-pro-type-member-init %t -- -- -fdelayed-template-parsing

template <class T>
struct PositiveFieldBeforeConstructor {
  int F;
  bool G /* with comment */;
  int *H;
  PositiveFieldBeforeConstructor() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: F, G, H
};
// Explicit instantiation.
template class PositiveFieldBeforeConstructor<int>;

template <class T>
struct PositiveFieldAfterConstructor {
  PositiveFieldAfterConstructor() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: F, G, H
  int F;
  bool G /* with comment */;
  int *H;
};
// Explicit instantiation.
template class PositiveFieldAfterConstructor<int>;
