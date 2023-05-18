// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp \
// RUN:     -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -emit-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp \
// RUN:     -fsyntax-only -verify -DIMPORT_MODULE_B

//--- records.h
struct A {
    int a;
    int b;
    int c;
};

struct NoNameEntity {
    struct {
        int a;
        int b;
        int c;
    };
};

union U {
    int a;
    int b;
    int c;
};

//--- another_records.h
struct A {
    int a;
    double b;
    float c;
};

struct NoNameEntity {
    struct {
        int a;
        unsigned b;
        long c;
    };
};

union U {
    int a;
    double b;
    short c;
};

//--- A.cppm
module;
#include "records.h"
export module A;
export using ::A;
export using ::NoNameEntity;
export using ::U;
export constexpr A a_a{};
export constexpr NoNameEntity NoName_a{};
export constexpr U u_a{};

//--- B.cppm
module;
#include "records.h"
export module B;
export using ::A;
export using ::NoNameEntity;
export using ::U;
export constexpr A a_b{};
export constexpr NoNameEntity NoName_b{};
export constexpr U u_b{};

//--- Use.cpp
// expected-no-diagnostics
import A;
#ifdef IMPORT_MODULE_B
import B;
static_assert(__is_same(decltype(a_a), decltype(a_b)));
static_assert(__is_same(decltype(NoName_a), decltype(NoName_b)));
static_assert(__is_same(decltype(u_a), decltype(u_b)));
#endif
void use1() {
    A a; // Shouldn't be ambiguous after import B;
    a.a = 43;
    a.b = 44;
    a.c = 45;
    NoNameEntity Anonymous;
    Anonymous.a = 46;
    Anonymous.b = 47;
    Anonymous.c = 48;
    U u;
    u.a = 43;
}
