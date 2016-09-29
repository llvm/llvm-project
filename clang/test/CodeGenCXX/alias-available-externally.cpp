// RUN: %clang_cc1 -O1 -std=c++11 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// Clang should not generate alias to available_externally definitions.
// Check that the destructor of Foo is defined.
// CHECK: define linkonce_odr void @_ZN3FooD2Ev
template <class CharT>
struct String {
  String() {}
  ~String();
};

template <class CharT>
inline __attribute__((visibility("hidden"), always_inline))
String<CharT>::~String() {}

extern template struct String<char>;

struct Foo : public String<char> { Foo() { String<char> s; } };

Foo f;
