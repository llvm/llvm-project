// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/P.cppm -emit-module-interface -o %t/M-P.pcm -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -emit-module-interface -o %t/M.pcm -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/M.cpp -fprebuilt-module-path=%t -fsyntax-only -verify

//--- A.cppm
export module A;
export using T = int;

//--- P.cppm
export module M:P;
import A;

//--- M.cppm
export module M;
export import :P;

//--- M.cpp
// expected-no-diagnostics
module M;

T x;
