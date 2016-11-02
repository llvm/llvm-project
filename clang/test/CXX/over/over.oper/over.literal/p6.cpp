// RUN: %clang_cc1 -std=c++11 %s -verify

// expected-note@+1 {{extern "C" language linkage specification begins here}}
extern "C" void operator "" _a(const char *); // expected-error {{must have C++ linkage}}
extern "C" template<char...> void operator "" _b(); // expected-error {{must have C++ linkage}}
// expected-note@-1 {{extern "C" language linkage specification begins here}}

extern "C" { // expected-note 4 {{extern "C" language linkage specification begins here}}
  void operator "" _c(const char *); // expected-error {{must have C++ linkage}}
  template<char...> void operator "" _d(); // expected-error {{must have C++ linkage}}
  namespace N {
    void operator "" _e(const char *); // expected-error {{must have C++ linkage}}
    template<char...> void operator "" _f(); // expected-error {{must have C++ linkage}}
  }
}
