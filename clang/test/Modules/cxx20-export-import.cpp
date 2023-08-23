
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/dummy.cppm -emit-module-interface -o %t/dummy.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -verify %t/test.cpp


//--- dummy.cppm
export module dummy;

//--- test.cpp
export import dummy; // expected-error {{export declaration can only be used within a module purview}}
