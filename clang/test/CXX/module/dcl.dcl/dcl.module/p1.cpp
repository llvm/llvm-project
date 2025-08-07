// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -verify %t/A.cppm
// RUN: %clang_cc1 -std=c++20 -verify %t/B.cppm
// RUN: %clang_cc1 -std=c++20 %t/C.cppm -emit-module-interface -o %t/C.pcm
// RUN: %clang_cc1 -std=c++20 %t/D.cppm -fmodule-file=foo=%t/C.pcm
// RUN: %clang_cc1 -std=c++20 %t/E.cppm -fmodule-file=foo=%t/C.pcm
// RUN: %clang_cc1 -std=c++20 -verify %t/F.cppm -fmodule-file=foo=%t/C.pcm

//--- A.cppm
export module foo; // expected-note {{previous module declaration is here}}
export module bar; // expected-error {{translation unit contains multiple module declarations}}

//--- B.cppm
export module foo; // expected-note {{previous module declaration is here}}
module bar; // expected-error {{translation unit contains multiple module declarations}}

//--- C.cppm
export module foo;

//--- D.cppm
module foo;

//--- E.cppm
export module bar;

//--- F.cppm
module foo; // expected-note {{previous module declaration is here}}
export module bar; // expected-error {{translation unit contains multiple module declarations}}
