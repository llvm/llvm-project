// RUN: rm -fr %t
// RUN: mkdir %t
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %S/Inputs/m8.cppm -I%S/Inputs -o %t/m8.pcm
// RUN: %clang_cc1 -std=c++20 -I%S/Inputs/ -fprebuilt-module-path=%t %s -verify -fsyntax-only
// expected-no-diagnostics
export module t8;
import m8;
