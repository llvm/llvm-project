// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs %s -verify
#include <Module/NotInModule.h> // expected-warning{{missing submodule 'Module.NotInModule'}}

int getNotInModule(void) {
  return not_in_module;
}
