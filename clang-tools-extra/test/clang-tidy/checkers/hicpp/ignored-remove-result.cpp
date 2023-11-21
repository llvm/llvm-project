// RUN: %check_clang_tidy %s hicpp-ignored-remove-result %t

namespace std {

template <typename ForwardIt, typename T>
ForwardIt remove(ForwardIt, ForwardIt, const T &);

template <typename ForwardIt, typename UnaryPredicate>
ForwardIt remove_if(ForwardIt, ForwardIt, UnaryPredicate);

template <typename ForwardIt>
ForwardIt unique(ForwardIt, ForwardIt);

template <class InputIt, class T>
InputIt find(InputIt, InputIt, const T&);

} // namespace std

void warning() {
  std::remove(nullptr, nullptr, 1);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning

  std::remove_if(nullptr, nullptr, nullptr);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning

  std::unique(nullptr, nullptr);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning
}

void noWarning() {

  auto RemoveRetval = std::remove(nullptr, nullptr, 1);

  auto RemoveIfRetval = std::remove_if(nullptr, nullptr, nullptr);

  auto UniqueRetval = std::unique(nullptr, nullptr);

  // cast to void should allow ignoring the return value
  (void)std::remove(nullptr, nullptr, 1);

  // no warning on std::find since the checker overrides
  // bugprone-unused-return-value's checked functions.
  std::find(nullptr, nullptr, 1);
}
