// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/foo.cppm -I%t -emit-module-interface -o %t/foo.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cc -fmodule-file=foo=%t/foo.pcm -fsyntax-only -verify

//--- foo.h
namespace foo {

}

//--- foo.cppm
module;
#include "foo.h"
export module foo;

//--- use.cc
import foo;
void use() {
    foo::bar(); // expected-error {{no member named 'bar' in namespace 'foo'}}
}
