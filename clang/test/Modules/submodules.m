
// RUN: rm -rf %t
// RUN: %clang_cc1 -Wauto-import -fmodule-cache-path %t -fmodules -F %S/Inputs %s -verify
// expected-no-diagnostics

// Note: transitively imports Module.Sub2.
@__experimental_modules_import Module.Sub;

int getValue() { 
  return *Module_Sub + *Module_Sub2;
}

