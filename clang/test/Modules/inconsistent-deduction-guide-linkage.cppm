// RUN: rm -fr %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -I%t -emit-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only %t/A.cppm -I%t -fprebuilt-module-path=%t -verify
//
// RUN: %clang_cc1 -std=c++20 %t/D.cppm -I%t -emit-module-interface -o %t/D.pcm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only %t/D-part.cppm -I%t -fprebuilt-module-path=%t -verify

// RUN: %clang_cc1 -std=c++20 %t/B.cppm -I%t -emit-reduced-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only %t/A.cppm -I%t -fprebuilt-module-path=%t -verify
//
// RUN: %clang_cc1 -std=c++20 %t/D.cppm -I%t -emit-reduced-module-interface -o %t/D.pcm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only %t/D-part.cppm -I%t -fprebuilt-module-path=%t -verify

//--- A.cppm
module;
export module baz:A;
import B;
#include "C.h"

//--- B.cppm
module;

#include "C.h"
export module B;

//--- C.h
namespace foo {
  template<class T, class U> struct bar { // expected-error {{declaration of 'bar' in module baz:A follows declaration in the global module}} // expected-note {{previous declaration is here}}
      template<class, class> bar(T, U);
  };
  template<class T, class U> bar(T, U) -> bar<T, U>; // expected-error {{declaration of '<deduction guide for bar>' in module baz:A follows declaration in the global module}} // expected-note {{previous declaration is here}}
}

//--- D.cppm
// Tests that it is still problematic if they are in one module.
module;
#include "E.h"
export module D;

//--- D-part.cppm
export module D:part;
import D;
#include "E.h"

//--- E.h
// another file for simpler diagnostics.
namespace foo {
  template<class T, class U> struct bar { // expected-error {{declaration of 'bar' in module D:part follows declaration in the global module}} // expected-note {{previous declaration is here}}
      template<class, class> bar(T, U);
  };
  template<class T, class U> bar(T, U) -> bar<T, U>; // expected-error {{declaration of '<deduction guide for bar>' in module D:part follows declaration in the global module}} // expected-note {{previous declaration is here}}
}
