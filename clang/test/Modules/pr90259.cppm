// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/mod1.cppm -emit-reduced-module-interface -o %t/mod-mod1.pcm
// RUN: %clang_cc1 -std=c++20 %t/mod.cppm -fprebuilt-module-path=%t  \
// RUN:     -emit-reduced-module-interface -o %t/mod.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fprebuilt-module-path=%t -verify -fsyntax-only

//--- mod1.cppm
export module mod:mod1;
namespace {
    int abc = 43;
}
namespace mod {
    static int def = 44;
}
export int f() {
    return abc + mod::def;
}

//--- mod.cppm
// expected-no-diagnostics
export module mod;
import :mod1;

namespace {
    double abc = 43.0;
}

namespace mod {
    static double def = 44.0;
}

export double func() {
    return (double)f() + abc + mod::def;
}

//--- use.cpp
// expected-no-diagnostics
import mod;
double use() {
    return func();
}
