// RUN: %clang_cc1 -ast-print -std=c++14 %s | FileCheck %s

namespace ns {

struct Wrapper {
class Inner {
  Inner();
  Inner(int);
  ~Inner();

  void operator += (int);

  template<typename T>
  void member();

  static void staticMember();

  operator int();

  operator ns::Wrapper();
  // CHECK: operator ns::Wrapper()
};
};

Wrapper::Inner::Inner() { }
// CHECK: Wrapper::Inner::Inner()

void Wrapper::Inner::operator +=(int) { }
// CHECK: void Wrapper::Inner::operator+=(int)

}

ns::Wrapper::Inner::Inner(int) { }
// CHECK: ns::Wrapper::Inner::Inner(int)

ns::Wrapper::Inner::~Inner() { }
// CHECK: ns::Wrapper::Inner::~Inner()

template<typename T>
void ::ns::Wrapper::Inner::member() { }
// CHECK: template <typename T> void ::ns::Wrapper::Inner::member()

ns::Wrapper::Inner::operator int() { return 0; }
// CHECK: ns::Wrapper::Inner::operator int()

ns::Wrapper::Inner::operator ::ns::Wrapper() { return ns::Wrapper(); }
// CHECK: ns::Wrapper::Inner::operator ::ns::Wrapper()

namespace ns {

void Wrapper::Inner::staticMember() { }
// CHECK: void Wrapper::Inner::staticMember()

}
