// RUN: %clang_cc1 -fsyntax-only  %s
struct foo {
  virtual void bar() ;
};
template<typename T>
class zed : public foo {
};
template<typename T>
class bah : public zed<T> {
  void f() {
     const_cast<foo *>(this->g())->bar();
  }
};
