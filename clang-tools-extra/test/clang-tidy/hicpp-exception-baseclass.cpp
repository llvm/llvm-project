// RUN: %check_clang_tidy %s hicpp-exception-baseclass %t -- -- -fcxx-exceptions

namespace std {
class exception {};
} // namespace std

class derived_exception : public std::exception {};
class non_derived_exception {};

void problematic() {
  try {
    throw int(42); // Built in is not allowed
// CHECK-MESSAGES: [[@LINE-1]]:5: warning: throwing an exception whose type is not derived from 'std::exception'
  } catch (int e) {
  }
  throw int(42); // Bad
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: throwing an exception whose type is not derived from 'std::exception'

  try {
    throw non_derived_exception(); // Some class is not allowed
// CHECK-MESSAGES: [[@LINE-1]]:5: warning: throwing an exception whose type is not derived from 'std::exception'
// CHECK-MESSAGES: 8:1: note: type defined here
  } catch (non_derived_exception &e) {
  }
  throw non_derived_exception(); // Bad
// CHECK-MESSAGES: [[@LINE-1]]:3: warning: throwing an exception whose type is not derived from 'std::exception'
// CHECK-MESSAGES: 8:1: note: type defined here
}

void allowed_throws() {
  try {
    throw std::exception(); // Ok
  } catch (std::exception &e) { // Ok
  }
  throw std::exception();

  try {
    throw derived_exception(); // Ok
  } catch (derived_exception &e) { // Ok
  }
  throw derived_exception(); // Ok
}
