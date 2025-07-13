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

void throwA() {
  throw std::runtime_error("ERROR A");
}

void throwB(int n) {
    if (n)
        throw std::runtime_error("ERROR B");
    else
        throw std::runtime_error("ERROR B with n=0");
}

int test(int x) {
    if (x > 0) 
        throwA(); 
    else 
        throwB(x); 
}
