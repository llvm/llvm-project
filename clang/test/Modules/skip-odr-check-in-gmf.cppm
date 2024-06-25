// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// Baseline testing to make sure we can detect the ODR violation from the CC1 invocation.
// RUNX: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUNX: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -o %t/b.pcm
// RUNX: %clang_cc1 -std=c++20 %t/test.cc -fprebuilt-module-path=%t -fsyntax-only -verify
//
// Testing that we can ignore the ODR violation from the driver invocation.
// RUN: %clang -std=c++20 %t/a.cppm --precompile -o %t/a.pcm
// RUN: %clang -std=c++20 %t/b.cppm --precompile -o %t/b.pcm
// RUN: %clang -std=c++20 %t/test.cc -fprebuilt-module-path=%t -fsyntax-only -Xclang -verify \
// RUN:     -DIGNORE_ODR_VIOLATION
//
// Testing that the driver can require to check the ODR violation.
// RUN: %clang -std=c++20 -Xclang -fno-skip-odr-check-in-gmf %t/a.cppm --precompile -o %t/a.pcm
// RUN: %clang -std=c++20 -Xclang -fno-skip-odr-check-in-gmf %t/b.cppm --precompile -o %t/b.pcm
// RUN: %clang -std=c++20 -Xclang -fno-skip-odr-check-in-gmf %t/test.cc -fprebuilt-module-path=%t \
// RUN:     -fsyntax-only -Xclang -verify

//--- func1.h
bool func(int x, int y) {
    return true;
}

//--- func2.h
bool func(int x, int y) {
    return false;
}

//--- a.cppm
module;
#include "func1.h"
export module a;
export using ::func;

//--- b.cppm
module;
#include "func2.h"
export module b;
export using ::func;

//--- test.cc
import a;
import b;
bool test() {
    return func(1, 2);
}

#ifdef IGNORE_ODR_VIOLATION
// expected-no-diagnostics
#else
// expected-error@func2.h:1 {{'func' has different definitions in different modules;}}
// expected-note@func1.h:1 {{but in 'a.<global>' found a different body}}
#endif
