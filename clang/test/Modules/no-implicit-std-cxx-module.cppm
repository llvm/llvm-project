// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/b.cppm -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/a.cppm -fmodule-file=b=%t/b.pcm \
// RUN:     -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/user.cpp -fmodule-file=a=%t/a.pcm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 %t/user.cpp -fmodule-file=a=%t/a.pcm -verify -fsyntax-only \
// RUN:     -Wno-read-modules-implicitly -DNO_DIAG 
// RUN: %clang_cc1 -std=c++20 %t/user.cpp -fmodule-file=a=%t/a.pcm -fmodule-file=b=%t/b.pcm \
// RUN:     -DNO_DIAG -verify -fsyntax-only

//--- b.cppm
export module b;
export int b() {
    return 43;
}

//--- a.cppm
export module a;
import b;
export int a() {
    return b() + 43;
}

//--- user.cpp
#ifdef NO_DIAG
// expected-no-diagnostics
#else
 // expected-warning@+2 {{it is deprecated to read module 'b' implcitly;}}
#endif
import a;
int use() {
    return a();
}