// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/a.cpp -fmodule-file=a=%t/a.pcm -ast-dump | FileCheck %s

// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/a.cpp -fmodule-file=a=%t/a.pcm -ast-dump | FileCheck %s

//--- a.h
namespace a {
class A {
public:
    int aaaa;

    int get() {
        return aaaa;
    }
};


template <class T>
class B {
public:
    B(T t): t(t) {}
    T t;
};

using BI = B<int>;

inline int get(A a, BI b) {
    return a.get() + b.t;
}

}

//--- a.cppm
export module a;

export extern "C++" {
#include "a.h"
}

//--- a.cpp
import a;
#include "a.h"

int test() {
    a::A aa;
    a::BI bb(43);
    return get(aa, bb);
}

// CHECK-NOT: DefinitionData
// CHECK: FunctionDecl {{.*}} get 'int (A, BI)' {{.*}}
// CHECK-NOT: CompoundStmt
// CHECK: FunctionDecl {{.*}} test {{.*}}
