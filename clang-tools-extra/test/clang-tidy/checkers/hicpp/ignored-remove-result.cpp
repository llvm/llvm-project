// RUN: %check_clang_tidy %s hicpp-ignored-remove-result %t
// RUN: %check_clang_tidy -check-suffixes=NOCAST %s hicpp-ignored-remove-result %t -- -config='{CheckOptions: {hicpp-ignored-remove-result.AllowCastToVoid: false}}'

namespace std {

template <typename ForwardIt, typename T>
ForwardIt remove(ForwardIt, ForwardIt, const T &);

template <typename ForwardIt, typename UnaryPredicate>
ForwardIt remove_if(ForwardIt, ForwardIt, UnaryPredicate);

template <typename ForwardIt>
ForwardIt unique(ForwardIt, ForwardIt);

template <class InputIt, class T>
InputIt find(InputIt, InputIt, const T&);

class error_code {
};

} // namespace std

std::error_code errorFunc() {
  return std::error_code();
}

void warning() {
  std::remove(nullptr, nullptr, 1);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning
  // CHECK-MESSAGES-NOCAST: [[@LINE-3]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors

  std::remove_if(nullptr, nullptr, nullptr);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning
  // CHECK-MESSAGES-NOCAST: [[@LINE-3]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors

  std::unique(nullptr, nullptr);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning
  // CHECK-MESSAGES-NOCAST: [[@LINE-3]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
}

void optionalWarning() {
  // No warning unless AllowCastToVoid=false
  (void)std::remove(nullptr, nullptr, 1);
  // CHECK-MESSAGES-NOCAST: [[@LINE-1]]:9: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
}

void noWarning() {

  auto RemoveRetval = std::remove(nullptr, nullptr, 1);

  auto RemoveIfRetval = std::remove_if(nullptr, nullptr, nullptr);

  auto UniqueRetval = std::unique(nullptr, nullptr);

  // Verify that other checks in the baseclass are not used.
  // - no warning on std::find since the checker overrides
  //   bugprone-unused-return-value's checked functions.
  std::find(nullptr, nullptr, 1);
  // - no warning on return types since the checker disable
  //   bugprone-unused-return-value's checked return types.
  errorFunc();
  (void) errorFunc();
}
