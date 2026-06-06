// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// Treat the behavior of using headers as baseline.
// RUN: %clang_cc1 -std=c++20 %t/use-header.cc -isystem %t -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -isystem %t -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/use-module.cc -isystem %t -fmodule-file=a=%t/a.pcm -fsyntax-only -verify

// Test again with reduced BMI.
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -isystem %t -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/use-module.cc -isystem %t -fmodule-file=a=%t/a.pcm -fsyntax-only -verify

//--- sys.h
#ifndef SYS_H
#define SYS_H

#pragma GCC system_header

template <class C>
struct [[deprecated]] iterator {};

_Pragma("GCC diagnostic push")
_Pragma("GCC diagnostic ignored \"-Wdeprecated\"")                         
_Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")

template <class C>
struct reverse_iterator 
: public iterator<C> {};

_Pragma("GCC diagnostic pop")

template <class T>
class C {
public:
    void i() {
        reverse_iterator<T> i;
    }
};

#endif

//--- use-header.cc
// expected-no-diagnostics
// However, we see unexpected warnings
#include <sys.h>

void use() {
    C<int>().i();
}

//--- a.cppm
module;
#include <sys.h>
export module a;
export using ::iterator;
export using ::C;

//--- use-module.cc
// expected-no-diagnostics
import a;

void use() {
    C<int>().i();
}
