
// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++20 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps -I%S/Inputs -verify %s
export import dummy; // expected-error {{export declaration can only be used within a module purview}}
