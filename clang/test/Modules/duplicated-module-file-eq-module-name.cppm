// Tests that we will pick the last `-fmodule-file=<module-name>=<path>` flag
// for <module-name>.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/u.cpp -fmodule-file=a=%t/unexist.pcm \
// RUN:      -fmodule-file=a=%t/a.pcm -verify -fsyntax-only

//--- a.cppm
export module a;
export int a();

//--- u.cpp
// expected-no-diagnostics
import a;
int u() {
    return a();
}
