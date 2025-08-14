// Tests referencing variable with initializer containing side effect across module boundary
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/Foo.cppm \
// RUN: -o %t/Foo.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface \
// RUN: -fmodule-file=Foo=%t/Foo.pcm \
// RUN: %t/Bar.cppm \
// RUN: -o %t/Bar.pcm

// RUN: %clang_cc1 -std=c++20 -emit-obj \
// RUN: -main-file-name Bar.cppm \
// RUN: -fmodule-file=Foo=%t/Foo.pcm \
// RUN: -x pcm %t/Bar.pcm \
// RUN: -o %t/Bar.o

//--- Foo.cppm
export module Foo;

export {
class S {};

inline S s = S{};
}

//--- Bar.cppm
export module Bar;
import Foo;

S bar() {
  return s;
}

