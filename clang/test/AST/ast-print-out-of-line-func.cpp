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

template<int x, typename T>
class TemplateRecord {
  void function();
  template<typename U> void functionTemplate(T, U);
};

template<int x, typename T>
void TemplateRecord<x, T>::function() { }
// CHECK: template <int x, typename T> void TemplateRecord<x, T>::function()

template<int x, typename T>
template<typename U>
void TemplateRecord<x, T>::functionTemplate(T, U) { }
// CHECK: template <int x, typename T> template <typename U> void TemplateRecord<x, T>::functionTemplate(T, U)

template<>
class TemplateRecord<0, int> {
  void function();
  template<typename U> void functionTemplate(int, U);
};

void TemplateRecord<0, int>::function() { }
// CHECK: void TemplateRecord<0, int>::function()

template<typename U>
void TemplateRecord<0, int>::functionTemplate(int, U) { }
// CHECK: template <typename U> void TemplateRecord<0, int>::functionTemplate(int, U)

template<typename T>
struct OuterTemplateRecord {
  template<typename U>
  struct Inner {
    void function();
  };
};

template<typename T>
template<typename U>
void OuterTemplateRecord<T>::Inner<U>::function() { }
// CHECK: template <typename T> template <typename U> void OuterTemplateRecord<T>::Inner<U>::function()
