// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/M-Part.cppm -emit-module-interface -o %t/M-Part.pcm
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -fmodule-file=M:Part=%t/M-Part.pcm -fsyntax-only -verify

//--- a.cppm
export module a;
export int foo() { return 43; }

//--- b.cppm
// expected-no-diagnostics
export module b;
import a;
export int b() {
    return foo();
}

//--- use.cpp
// expected-no-diagnostics
import a;
int Use() {
    return foo();
}

//--- M-Part.cppm
export module M:Part;

//--- M.cppm
// expected-no-diagnostics
export module M;
import :Part;
