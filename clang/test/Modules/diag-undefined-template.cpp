// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++20 -fmodules -fmodules-cache-path=%t -I%S/Inputs/undefined-template \
// RUN:   -Wundefined-func-template \
// RUN:   -fimplicit-module-maps  %s 2>&1 | grep "declaration of template entity is unreachable here"

#include "hoge.h"

int main() {
  f();
}
