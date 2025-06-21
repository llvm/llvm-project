// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -fexceptions -Wreturn-type -verify %s
// expected-no-diagnostics

namespace std {
  class string {
  public:
    string(const char*); // constructor for runtime_error
  };
  class runtime_error {
  public:
    runtime_error(const string &); 
  };
}

void throwError(const std::string& msg) {
  throw std::runtime_error(msg);
}

int ensureZero(const int i) {
  if (i == 0) return 0;
  throwError("ERROR"); // no-warning
}

int alwaysThrows() {
  throw std::runtime_error("This function always throws"); // no-warning
}
