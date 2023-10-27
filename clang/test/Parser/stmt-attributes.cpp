// RUN: %clang_cc1 -std=c++11 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s

#if !__has_extension(statement_attributes_with_gnu_syntax)
#error "We should have statement attributes with GNU syntax support"
#endif

template <typename T = void>
class __attribute__((nomerge)) A {
  // expected-error@-1 {{'nomerge' attribute only applies to functions, statements and variables}}
};

class B : public A<> {
public:
  void bar();
};

void bar();

void foo(A<> *obj) {
  __attribute__((nomerge)) static_cast<B *>(obj)->bar();
  __attribute__((nomerge))[obj]() { static_cast<B *>(obj)->bar(); }
  ();
  __attribute__(()) try {
    bar();
  } catch (...) {
  }
}
