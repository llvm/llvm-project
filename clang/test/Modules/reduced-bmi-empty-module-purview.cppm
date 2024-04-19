// Test that we won't write additional information into the Reduced BMI if the 
// module purview is empty.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -emit-reduced-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-reduced-module-interface -o %t/A.pcm \
// RUN:     -fmodule-file=M=%t/M.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram --show-binary-blobs %t/A.pcm > %t/A.dump
// RUN: cat %t/A.dump | FileCheck %t/A.cppm
//
// RUN: %clang_cc1 -std=c++20 %t/A1.cppm -emit-reduced-module-interface -o %t/A1.pcm \
// RUN:     -fmodule-file=M=%t/M.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram --show-binary-blobs %t/A1.pcm > %t/A1.dump
// RUN: cat %t/A1.dump | FileCheck %t/A1.cppm

//--- foo.h
namespace ns {
template <class C>
class A {

};

extern template class A<short>;

inline A<int> a() { return A<int>(); }
template <class T>
A<T> _av_ = A<T>();

auto _av_1 = _av_<int>;
auto _av_2 = _av_<double>;

template <>
class A<void> {

};

void func(A<int>, ...) {

}

}

struct S {
    union {
        unsigned int V;
        struct {
            int v1;
            int v2;
            ns::A<int> a1;
        } WESQ;
    };

    union {
        double d;
        struct {
            int v1;
            unsigned v2;
            ns::A<unsigned> a1;
        } Another;
    };
};

//--- M.cppm
module;
#include "foo.h"
export module M;
export namespace nv {
    using ns::A;
    using ns::a;
    using ns::_av_;

    using ns::func;
}
using ::S;

//--- A.cppm
module;
#include "foo.h"
export module A;
import M;

// CHECK-NOT: <DECL_CXX_RECORD
// CHECK-NOT: <DECL_UPDATE_OFFSETS

//--- A1.cppm
module;
import M;
#include "foo.h"
export module A;

// CHECK-NOT: <DECL_CXX_RECORD
// CHECK-NOT: <DECL_UPDATE_OFFSETS

