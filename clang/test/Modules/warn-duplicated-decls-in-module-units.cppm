// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/m1.cppm -emit-module-interface -o %t/m1.pcm
// RUN: %clang_cc1 -std=c++20 %t/m2.cppm -emit-module-interface -o %t/m2.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/use.cc -fsyntax-only \
// RUN:     -verify
//
// RUN: %clang_cc1 -std=c++20 %t/m1.cppm -Wall -emit-module-interface -o %t/m1.pcm
// RUN: %clang_cc1 -std=c++20 %t/m2.cppm -Wall -emit-module-interface -o %t/m2.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/use.cc -fsyntax-only \
// RUN:     -verify -Wall
//
// RUN: %clang_cc1 -std=c++20 %t/m1.cppm -Wdecls-in-multiple-modules -emit-module-interface -o %t/m1.pcm
// RUN: %clang_cc1 -std=c++20 %t/m2.cppm -Wdecls-in-multiple-modules -emit-module-interface -o %t/m2.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/use.cc -fsyntax-only \
// RUN:     -verify -Wdecls-in-multiple-modules -DWARNING

//--- foo.h
#ifndef FOO_H
#define FOO_H

enum E { E1, E2 };

int a = 43;

class foo {
public:
    void consume(E, int);
};

inline void func() {}

void fwd_decl();

#endif 

//--- m1.cppm
module;
#include "foo.h"
export module m1;
export {
    using ::foo;
    using ::a;
    using ::func;
    using ::fwd_decl;
    using ::E;
}

//--- m2.cppm
module;
#include "foo.h"
export module m2;
export {
    using ::foo;
    using ::a;
    using ::func;
    using ::fwd_decl;
    using ::E;
}

//--- use.cc
import m1;
import m2;
void use();
void use() {
    E e = E1;
    foo f;
    f.consume(e, a);
    func();
    fwd_decl();
}

#ifndef WARNING
// expected-no-diagnostics
#else
// expected-warning@* {{declaration 'E' is detected to be defined in multiple module units}}
// expected-warning@* {{declaration 'foo' is detected to be defined in multiple module units}}
// expected-warning@* {{declaration 'a' is detected to be defined in multiple module units}}
// expected-warning@* {{declaration 'func' is detected to be defined in multiple module units}}
// expected-note@* 1+ {{}}
#endif
