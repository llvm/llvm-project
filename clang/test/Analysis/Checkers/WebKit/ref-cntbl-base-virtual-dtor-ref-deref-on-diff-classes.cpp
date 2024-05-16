// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.RefCntblBaseVirtualDtor -verify %s

struct RefCountedBase {
  void ref() {}
};

template<typename T> struct RefCounted : RefCountedBase {
public:
  void deref() const { }
};

struct Base : RefCounted<Base> {
// expected-warning@-1{{Struct 'RefCounted<Base>' is used as a base of struct 'Base' but doesn't have virtual destructor}}
  virtual void foo() { }
};

struct Derived : Base { };
// expected-warning@-1{{Struct 'Base' is used as a base of struct 'Derived' but doesn't have virtual destructor}}

void foo () {
  Derived d;
}
