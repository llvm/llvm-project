// RUN: %clang_cc1 -fsyntax-only -verify %s

int ovl(int); // expected-note 3{{possible target for call}}
float ovl(float); // expected-note 3{{possible target for call}}

template<typename T> T ovl(T); // expected-note 3{{possible target for call}}

void test(bool b) {
  (void)((void)0, ovl); // expected-error{{reference to overloaded function could not be resolved; did you mean to call it?}}
  // PR7863
  (void)(b? ovl : &ovl); // expected-error{{reference to overloaded function could not be resolved; did you mean to call it?}}
  (void)(b? ovl<float> : &ovl); // expected-error{{reference to overloaded function could not be resolved; did you mean to call it?}}
  (void)(b? ovl<float> : ovl<float>);
}

namespace rdar9623945 {
  void f(...) {
  }
  
  class X {
  public:
    const char* text(void);
    void g(void) {
      f(text());
      f(text); // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
      f(text());
      f(text); // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
    }
  };
}
