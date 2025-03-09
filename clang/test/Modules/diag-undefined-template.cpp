// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++20 -fmodules -fmodules-cache-path=%t -I%S/Inputs/undefined-template \
// RUN:   -Wundefined-func-template \
// RUN:   -fimplicit-module-maps  %s 2>&1 | grep "unreachable declaration of template entity is her"

#include "hoge.h"

int main() {
  f();
}
