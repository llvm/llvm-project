// REQUIRES: shell

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t/modules -fmodules -fimplicit-module-maps -I %S/Inputs -emit-pch -o %t.pch %s -verify

// RUN: ln -s %t/modules %t/modules.symlink
// RUN: %clang_cc1 -fmodules-cache-path=%t/modules.symlink -fmodules -fimplicit-module-maps -I %S/Inputs -include-pch %t.pch %s -verify
// RUN: not %clang_cc1 -fmodules-cache-path=%t/modules.dne -fmodules -fimplicit-module-maps -I %S/Inputs -include-pch %t.pch %s -verify

// expected-no-diagnostics

@import ignored_macros;

struct Point p;
