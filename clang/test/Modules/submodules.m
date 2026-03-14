
// RUN: rm -rf %t
// RUN: %clang_cc1 -Rmodule-include-translation -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs %s -verify
// expected-no-diagnostics

// Note: transitively imports Module.Sub2.
@import Module.Sub;

int getValue(void) { 
  return *Module_Sub + *Module_Sub2;
}

