// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
// PR5697
namespace PR5697 {
struct A {
  virtual void f() { } 
  A();
  A(int);
};

// A does not have a key function, so the first constructor we emit should
// cause the vtable to be defined (without assertions.)
// CHECK: @_ZTVN6PR56971AE = linkonce_odr unnamed_addr constant
A::A() { }
A::A(int) { }
}

// Make sure that we don't assert when building the vtable for a class
// template specialization or explicit instantiation with a key
// function.
template<typename T>
struct Base {
  virtual ~Base();
};

template<typename T>
struct Derived : public Base<T> { };

template<>
struct Derived<char> : public Base<char> {
  virtual void anchor();
};

void Derived<char>::anchor() { }
