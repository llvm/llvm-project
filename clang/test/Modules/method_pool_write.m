// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -fsyntax-only -I %S/Inputs %s -verify
// expected-no-diagnostics

@import MethodPoolCombined;
@import MethodPoolString2;

void message_kindof_object(__kindof S2 *kindof_S2) {
  [kindof_S2 stringValue];
}

