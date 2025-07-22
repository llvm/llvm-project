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

// Non-template version.

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

// Template version.

template<typename T> 
void throwErrorTemplate(const T& msg) {
  throw msg;
}

template <typename T>
int ensureZeroTemplate(T i) {
  if (i == 0) return 0;
  throwErrorTemplate("ERROR"); // no-warning
}

void testTemplates() {
  throwErrorTemplate("ERROR");
  (void)ensureZeroTemplate(42);
}
