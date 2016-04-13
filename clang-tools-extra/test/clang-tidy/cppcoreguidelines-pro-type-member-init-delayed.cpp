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

// This declaration isn't used and won't be parsed 'delayed-template-parsing'.
// The body of the declaration is 'null' and may cause crash if not handled
// properly by checkers.
template <class T>
struct UnusedDelayedConstructor {
  UnusedDelayedConstructor() {}
  int F;
};
