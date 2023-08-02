// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -I%t %t/module1.cppm -o %t/module1.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -I%t %t/module2.cppm -o %t/module2.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/merge.cpp -verify -fsyntax-only

//--- header.h
namespace NS {
template <int I>
class A {
};

template <template <int I_> class T>
class B {
};
}

//--- module1.h
namespace NS {
using C = B<A>;
}
struct D : NS::C {
    using Type = NS::C;
};

//--- module1.cppm
// inside NS, using C = B<A>
module;
#include "header.h"
#include "module1.h"
export module module1;
export using ::D;

//--- module2.h
namespace NS {
using C = B<NS::A>;
}
struct D : NS::C {
    using Type = NS::C;
};

//--- module2.cppm
// inside NS, using C = B<NS::A>
module;
#include "header.h"
#include "module2.h"
export module module2;
export using ::D;

//--- merge.cpp
// expected-no-diagnostics
import module1;
import module2;
D d;
