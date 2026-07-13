// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/m-a.cppm -emit-module-interface -o %t/m-a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -o %t/m-b.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/b.cpp -fsyntax-only -verify -fprebuilt-module-path=%t

//--- a.cppm
export module a;
namespace aa {

}

//--- m-a.cppm
export module m:a;
namespace aa {
struct A {};
}

//--- b.cppm
module m:b;
import :a;

namespace bb {
struct B {
    void func(aa::A);
};
}

//--- b.cpp
// expected-no-diagnostics
module m:b.impl;
import a;
import :b;

namespace bb {
using namespace aa;

void B::func(A) {} 

}



