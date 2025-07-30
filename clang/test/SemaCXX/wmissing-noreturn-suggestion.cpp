// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -fexceptions -Wreturn-type -Wmissing-noreturn -verify %s

namespace std {
  class string {
  public:
    string(const char*);
  };
  class runtime_error {
  public:
    runtime_error(const string&);
  };
}

// This function always throws. Suggest [[noreturn]].
void throwError(const std::string& msg) { // expected-warning {{function 'throwError' could be declared with attribute 'noreturn'}}
  throw std::runtime_error(msg);
}

// The non-void caller should not warn about missing return.
int ensureZero(int i) {
  if (i == 0) return 0;
  throwError("ERROR"); // no-warning
}


template <typename Ex>
[[noreturn]]
void tpl_throws(Ex const& e) {
    throw e;
}

[[noreturn]]
void tpl_throws_test() {
    tpl_throws(0);
}

[[gnu::noreturn]]
int gnu_throws() {
    throw 0;
}

[[noreturn]]
int cxx11_throws() {
    throw 0;
}
