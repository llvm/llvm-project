// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/inferred-attr -fsyntax-only -verify %s
// expected-no-diagnostics
extern "C" {
@import InferredExternC;
}
