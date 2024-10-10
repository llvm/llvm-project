// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -verify %t/A.cpp
// RUN: %clang_cc1 -std=c++20 -verify %t/B.cpp
// RUN: %clang_cc1 -std=c++20 %t/C.cpp -emit-module-interface -o %t/C.pcm
// RUN: %clang_cc1 -std=c++20 %t/D.cpp -fmodule-file=foo=%t/C.pcm
// RUN: %clang_cc1 -std=c++20 %t/E.cpp -fmodule-file=foo=%t/C.pcm
// RUN: %clang_cc1 -std=c++20 -verify %t/F.cpp -fmodule-file=foo=%t/C.pcm

//--- A.cpp
module;
export module foo; // expected-note {{previous module declaration is here}}
export module bar; // expected-error {{translation unit contains multiple module declarations}}

//--- B.cpp
module;
export module foo;  // expected-note {{previous module declaration is here}}
module bar;         // expected-error {{translation unit contains multiple module declarations}}

//--- C.cpp
module;
export module foo;

//--- D.cpp
module;
module foo;

//--- E.cpp
module;
export module bar;

//--- F.cpp
module;
module foo; // expected-note {{previous module declaration is here}}
export module bar; // expected-error {{translation unit contains multiple module declarations}}
