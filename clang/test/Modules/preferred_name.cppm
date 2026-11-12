// Tests that the ODR check wouldn't produce false-positive result for preferred_name attribute.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -I%t %t/Use.cppm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -I%t %t/Use1.cpp -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -I%t %t/Use2.cpp -verify -fsyntax-only

// Test again with reduced BMI.
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-reduced-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -I%t %t/Use.cppm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -I%t %t/Use1.cpp -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -I%t %t/Use2.cpp -verify -fsyntax-only
//
//--- foo.h
template<class _CharT>
class foo_templ;

typedef foo_templ<char> foo;

template<class _CharT>
class
__attribute__((__preferred_name__(foo)))
foo_templ {
public:
    foo_templ() {}
};

inline foo_templ<char> bar()
{
  return foo_templ<char>();
}

//--- A.cppm
module;
#include "foo.h"
export module A;
export using ::foo_templ;

//--- Use.cppm
// expected-no-diagnostics
module;
#include "foo.h"
export module Use;
import A;
export using ::foo_templ;

//--- Use1.cpp
import A;         // expected-warning@foo.h:8 {{attribute declaration must precede definition}}
#include "foo.h"  // expected-note@foo.h:9 {{previous definition is here}}

//--- Use2.cpp
// expected-no-diagnostics
#include "foo.h"
import A;
