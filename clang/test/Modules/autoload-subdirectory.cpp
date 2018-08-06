// RUN: %clang_cc1 -fmodules -fmodule-name=Foo -I %S/Inputs/autoload-subdirectory/ %s -verify
// expected-no-diagnostics

#include "a.h"
#import "c.h"

int main() {
  foo neko;
  return 0;
}
