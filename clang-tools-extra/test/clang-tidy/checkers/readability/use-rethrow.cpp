// RUN: %check_clang_tidy %s readability-use-rethrow %t

namespace std {
class exception {
public:
  virtual ~exception() = default;
  virtual const char* what() const { return "exception"; }
};
} // namespace std

void log(const char*);
void f();

void test_const_reference() {
  try {
    f();
  } catch (const std::exception &e) {
    log(e.what());
    throw e;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught exception; use a bare 'throw' to rethrow the original exception object [readability-use-rethrow]
    // CHECK-FIXES: throw;
  }
}

void test_non_const_reference() {
  try {
    f();
  } catch (std::exception &e) {
    throw e;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught exception; use a bare 'throw' to rethrow the original exception object [readability-use-rethrow]
    // CHECK-FIXES: throw;
  }
}

void test_catch_by_value() {
  try {
    f();
  } catch (int i) {
    i = 67;
    // We should NOT warn here.
    throw i; 
  } catch (std::exception e) {
    // Should NOT warn here either, as 'e' was caught by value.
    throw e; 
  }
}

void test_unrelated_throw() {
  try {
    f();
  } catch (const std::exception &e) {
    std::exception other;
    // Should NOT warn because we are throwing a different object.
    throw other;
  }
}
