// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/use.cc -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/use.cc -fmodule-file=a=%t/a.pcm -emit-llvm -o - | \
// RUN:     FileCheck %t/use.cc

//--- a.cppm
export module a;
namespace {
struct Local {};
}

export class A { 
public:
    void *external_but_not_type_external(Local *) {
        return nullptr;
    }
};

//--- use.cc
// expected-no-diagnostics
import a;
void *use() {
    A a;
    return a.external_but_not_type_external(nullptr);
}

// CHECK: define {{.*}}internal {{.*}}@_ZNW1a1A30external_but_not_type_externalEPN12_GLOBAL__N_15LocalE
