// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:    -working-directory %S/Inputs \
// RUN:    -I subdirectory-module-maps-working-dir \
// RUN:    %s -Werror=implicit-function-declaration -verify

@import ModuleInSubdir;

void foo(void) {
  int x = bar();
}

// expected-no-diagnostics
