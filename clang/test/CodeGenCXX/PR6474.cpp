// RUN: %clang_cc1 %s -emit-llvm-only

namespace test0 {
template <typename T> struct X {
  virtual void foo();
  virtual void bar();
  virtual void baz();
};

template <typename T> void X<T>::foo() {}
template <typename T> void X<T>::bar() {}
template <typename T> void X<T>::baz() {}

template <> void X<char>::foo() {}
template <> void X<char>::bar() {}
}

namespace test1 {
template <typename T> struct X {
  virtual void foo();
  virtual void bar();
  virtual void baz();
};

template <typename T> void X<T>::foo() {}
template <typename T> void X<T>::bar() {}
template <typename T> void X<T>::baz() {}

template <> void X<char>::bar() {}
template <> void X<char>::foo() {}
}
