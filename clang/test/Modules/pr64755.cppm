// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a0 -x c++ -emit-module %t/module.modulemap -o %t/a0.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fmodule-file=%t/a0.pcm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fmodule-file=a0=%t/a0.pcm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fprebuilt-module-path=%t -verify -fsyntax-only

//--- module.modulemap
module a0 { header "a0.h" export * }

//--- a0.h
void a0() {}

//--- use.cpp
import a0; // expected-error {{import of module 'a0' imported non C++20 importable modules}}
