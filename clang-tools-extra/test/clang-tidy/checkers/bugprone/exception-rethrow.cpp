// RUN: %check_clang_tidy %s bugprone-exception-rethrow %t -- -- -fexceptions

struct exception {};

namespace std {
  template <class T>
  T&& move(T &x) {
    return static_cast<T&&>(x);
  }
}

void correct() {
  try {
      throw exception();
  } catch(const exception &) {
      throw;
  }
}

void correct2() {
  try {
      throw exception();
  } catch(const exception &e) {
      try {
        throw exception();
      } catch(...) {}
  }
}

void not_correct() {
  try {
      throw exception();
  } catch(const exception &e) {
      throw e;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: throwing a copy of the caught 'exception' exception, remove the argument to throw the original exception object [bugprone-exception-rethrow]
  }
}

void not_correct2() {
  try {
      throw exception();
  } catch(const exception &e) {
      throw (e);
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: throwing a copy of the caught 'exception' exception, remove the argument to throw the original exception object [bugprone-exception-rethrow]
  }
}

void not_correct3() {
  try {
      throw exception();
  } catch(const exception &e) {
      throw exception(e);
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: throwing a copy of the caught 'exception' exception, remove the argument to throw the original exception object [bugprone-exception-rethrow]
  }
}

void not_correct4() {
  try {
      throw exception();
  } catch(exception &e) {
      throw std::move(e);
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: throwing a copy of the caught 'exception' exception, remove the argument to throw the original exception object [bugprone-exception-rethrow]
  }
}

void not_correct5() {
  try {
      throw 5;
  } catch(const int &e) {
      throw e;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: throwing a copy of the caught 'int' exception, remove the argument to throw the original exception object [bugprone-exception-rethrow]
  }
}

void rethrow_not_correct() {
  throw;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: empty 'throw' outside a catch block with no operand triggers 'std::terminate()' [bugprone-exception-rethrow]
}

void rethrow_not_correct2() {
  try {
    throw;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: empty 'throw' outside a catch block with no operand triggers 'std::terminate()' [bugprone-exception-rethrow]
  } catch(...) {
  }
}

void rethrow_correct() {
  try {
    throw 5;
  } catch(...) {
    throw;
  }
}

void rethrow_in_lambda() {
  try {
    throw 5;
  } catch(...) {
    auto lambda = [] { throw; };
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: empty 'throw' outside a catch block with no operand triggers 'std::terminate()' [bugprone-exception-rethrow]
  }
}
