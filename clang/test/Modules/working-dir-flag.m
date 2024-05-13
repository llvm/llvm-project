// RUN: rm -rf %t.mcp
// RUN: %clang_cc1 -fmodules-cache-path=%t.mcp -fmodules -fimplicit-module-maps -F . -working-directory=%S/Inputs/working-dir-test %s -verify
// expected-no-diagnostics

@import Test;

void foo(void) {
  test_me_call();
}
