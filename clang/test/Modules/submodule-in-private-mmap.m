// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F%S/Inputs/submodule-in-private-mmap -fsyntax-only %s -Wno-private-module -verify

// expected-no-diagnostics

@import A.Private;
@import A.SomeSub;
