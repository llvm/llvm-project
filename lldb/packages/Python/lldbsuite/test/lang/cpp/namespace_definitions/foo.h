//===-- foo.h ---------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class ThingInside;

namespace Foo {
  class MyClass {
  ThingInside *thing;
  public:
    MyClass() { }
  };
}
