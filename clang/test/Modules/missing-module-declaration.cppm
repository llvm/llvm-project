// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -I%t -emit-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -I%t -fprebuilt-module-path=%t -emit-module-interface -verify

//--- A.cppm
import B; // expected-error{{missing 'export module' declaration in module interface unit}}

//--- B.cppm
module;
export module B;
