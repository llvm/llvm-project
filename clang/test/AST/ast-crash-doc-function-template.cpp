// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -x c++ -Wdocumentation -fsyntax-only -ast-dump-all %t/t.cpp

//--- t.h
/// MyClass in the header file
class MyClass {
public:
  template <typename T>
  void Foo() const;

  /// Bar
  void Bar() const;
};

//--- t.cpp
#include "t.h"

/// MyClass::Bar: Foo<int>() is implicitly instantiated and called here.
void MyClass::Bar() const {
  Foo<int>();
}

/// MyClass::Foo
template <typename T>
void MyClass::Foo() const {
}

// CHECK: TranslationUnitDecl
