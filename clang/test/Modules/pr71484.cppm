// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/mod1.cppm -emit-module-interface -o %t/mod1.pcm
// RUN: %clang_cc1 -std=c++20 %t/mod2.cppm -emit-module-interface -o %t/mod2.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN:  %clang_cc1 -std=c++20 %t/mod3.cppm -fsyntax-only -verify \
// RUN:     -fprebuilt-module-path=%t

// RUN: %clang_cc1 -std=c++20 %t/mod1.cppm -emit-obj -o %t/mod1.o -fmodule-output=%t/mod1.pcm
// RUN: %clang_cc1 -std=c++20 %t/mod2.cppm -emit-obj -o %t/mod2.o -fmodule-output=%t/mod2.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN:  %clang_cc1 -std=c++20 %t/mod3.cppm -fsyntax-only -verify \
// RUN:     -fprebuilt-module-path=%t

//--- mod1.cppm
export module mod1;

export template<class T>
T mod1_f(T x) {
    return x;
}

//--- mod2.cppm
export module mod2;
import mod1;

export template<class U>
U mod2_g(U y) {
    return mod1_f(y);
}

//--- mod3.cppm
// expected-no-diagnostics
export module mod3;
import mod2;

export int mod3_h(int p) {
    return mod2_g(p);
}
