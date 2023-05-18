// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

enum class EC { ec1 };

void test0(EC ec) {
  (void)static_cast<bool>(ec);
  (void)static_cast<bool>(EC::ec1);
  (void)static_cast<char>(ec);
  (void)static_cast<char>(EC::ec1);
  (void)static_cast<int>(ec);
  (void)static_cast<int>(EC::ec1);
  (void)static_cast<unsigned long>(ec);
  (void)static_cast<unsigned long>(EC::ec1);
  (void)static_cast<float>(ec);
  (void)static_cast<float>(EC::ec1);
  (void)static_cast<double>(ec);
  (void)static_cast<double>(EC::ec1);
}

namespace PR9107 {
  enum E {};
  template <class _Tp> inline _Tp* addressof(_Tp& __x) {
    return (_Tp*)&(char&)__x;
  }
  void test() {
    E a;
    addressof(a);
  }
}
