// RUN: %check_clang_tidy %s bugprone-exception-rethrow %t -- -- -fexceptions

struct exception {};

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
      throw 5;
  } catch(const int &e) {
      throw e;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: throwing a copy of the caught 'int' exception, remove the argument to throw the original exception object [bugprone-exception-rethrow]
  }
}

void rethrow_not_correct() {
  throw;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: empty 'throw' outside a catch block without an exception can trigger 'std::terminate' [bugprone-exception-rethrow]
}

void rethrow_not_correct2() {
  try {
    throw;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: empty 'throw' outside a catch block without an exception can trigger 'std::terminate' [bugprone-exception-rethrow]
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
