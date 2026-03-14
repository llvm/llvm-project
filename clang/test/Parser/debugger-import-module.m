// RUN: %clang_cc1 -fsyntax-only -fdebugger-support -verify %s
// expected-no-diagnostics

void importAModule(void) {
  @import AModuleThatDoesntExist
}
