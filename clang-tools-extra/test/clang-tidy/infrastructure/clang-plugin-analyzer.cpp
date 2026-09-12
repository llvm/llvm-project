// REQUIRES: clang-plugin, static-analyzer
// UNSUPPORTED: system-windows
// RUN: %clang_cc1 -load %llvmshlibdir/clangTidyPlugin%pluginext -add-plugin clang-tidy -plugin-arg-clang-tidy -checks=-*,clang-analyzer-core.DivideZero %s -verify

int divide() {
  return 1 / 0; // expected-warning {{division by zero is undefined}}
                // expected-warning@-1 {{Division by zero [clang-analyzer-core.DivideZero]}}
                // expected-note@-2 {{Division by zero}}
}
