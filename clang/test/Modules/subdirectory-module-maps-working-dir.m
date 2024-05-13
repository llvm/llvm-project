// RUN: rm -rf %t
// RUN: %clang -fsyntax-only -fmodules -fmodules-cache-path=%t \
// RUN:    -working-directory %S/Inputs \
// RUN:    -I subdirectory-module-maps-working-dir \
// RUN:    %s -Werror=implicit-function-declaration -Xclang -verify

@import ModuleInSubdir;

void foo(void) {
  int x = bar();
}

// expected-no-diagnostics
