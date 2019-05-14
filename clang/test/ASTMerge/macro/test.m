// RUN: rm -rf %t
// RUN: mkdir -p %t/cache
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fmodule-map-file=%S/Inputs/macro.modulemap -I%S/Inputs -emit-pch -o %t.1.ast %S/Inputs/macro1.m
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fmodule-map-file=%S/Inputs/macro.modulemap -I%S/Inputs -emit-pch -o %t.2.ast %S/Inputs/macro2.m
// RUN: %clang_cc1 -fmodules -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only -verify %s
// expected-no-diagnostics
