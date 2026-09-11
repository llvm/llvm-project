// RUN: %clang_cc1 %s -triple=i386-pc-win32 -std=c++23 -fms-extensions -verify

consteval int bad(int x) { return 42 / x; } // expected-note{{division by zero}}

struct ExportedDefaultArgClosure {
  __declspec(dllexport)                 // expected-note{{in the default initializer of 'x'}}
  ExportedDefaultArgClosure(int x       // expected-note{{declared here}}
                            = bad(0)) { // expected-error{{call to consteval function 'bad' is not a constant expression}} expected-note{{in call to 'bad(0)'}}
  }
};

namespace GH67685 {
class Test1 {
public:
  int a;
};

consteval int f1() {
  Test1 var;
  var.Test1::Test1(var); // expected-warning {{explicit constructor calls are a Microsoft extension}}
  return 1;
}

static_assert(f1());
}
