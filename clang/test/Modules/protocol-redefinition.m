// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F%S/Inputs/protocol-redefinition -fsyntax-only %s -Wno-private-module -verify

// expected-no-diagnostics

@import Kit;
