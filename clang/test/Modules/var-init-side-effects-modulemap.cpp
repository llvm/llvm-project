// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -fsyntax-only -fmodules -fmodules-cache-path=%t -fmodule-map-file=%t/module.modulemap  %t/test.cppm -I%t
//

//--- test.cppm
#pragma clang module import Baz

//--- Foo.h
#pragma once
class foo {
  char dummy = 1;

public:
  static foo var;

};

inline foo foo::var;

//--- Bar.h
#pragma once
#include <Foo.h>

void bar() {
  (void) foo::var;
}

//--- Baz.h
#pragma once
#include <Foo.h>

void baz() {
  (void) foo::var;
}

#include <Bar.h>

//--- module.modulemap
module Foo {
  header "Foo.h"
}
module Bar {
  header "Bar.h"
}
module Baz {
  header "Baz.h"
}

