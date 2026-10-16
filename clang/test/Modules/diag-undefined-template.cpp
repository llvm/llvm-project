// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++20 -fmodules -fmodules-cache-path=%t -I%t \
// RUN:   -Wundefined-func-template \
// RUN:   -fimplicit-module-maps %t/main.cpp 2>&1 | grep "unreachable declaration of template entity is here"

// Note that the diagnostics are triggered when building the 'hoge' module, which is imported from the main file.
// The "-verify" flag doesn't work in this case. Instead, we grep the expected text to verify the test.

//--- shared_ptr2.h
#pragma once

template<class T>
void x() { }

//--- hoge.h
#pragma once

#include "shared_ptr2.h"

inline void f() {
  x<int>();
}

//--- module.modulemap
module hoge {
  header "hoge.h"
}

module shared_ptr2 {
  header "shared_ptr2.h"
}

//--- main.cpp
#include "hoge.h"

int main() {
  f();
}
