// Tests that the namespace with abi tag attribute won't get discarded.
// The pattern is widely used in libstdc++.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/m.cppm -emit-module-interface -o %t/m.pcm
// RUNX: %clang_cc1 -std=c++20 %t/m.pcm -S -emit-llvm -o - | FileCheck %t/m.cppm

//--- foo.h
#pragma GCC system_header

namespace std {
    inline namespace __cxx11 __attribute__((__abi_tag__ ("tag_name"))) { }
}

namespace __gnu_cxx {
    inline namespace __cxx11 __attribute__((__abi_tag__ ("tag_name"))) { }
}

namespace std {
    namespace __cxx11 {
        struct C { int x; };
    }
}

std::C foo() { return {3}; }

//--- m.cppm
module;
#include "foo.h"
export module m;
export using ::foo;

// CHECK: define{{.*}}@_Z3fooB8tag_namev

//--- bar.h
#pragma GCC system_header

namespace std {
    inline namespace __cxx11 __attribute__((__abi_tag__ ("tag_name"))) { }
}

namespace __gnu_cxx {
    inline namespace __cxx11 __attribute__((__abi_tag__ ("tag_name"))) { }
}

namespace __gnu_cxx {
    namespace __cxx11 {
        template <class C>
        struct traits {
            typedef C type;
        };
    }
}

namespace std {
    template <class C>
    struct vec {
        typedef traits<C>::type type;
    };
}

//--- n.cppm
module;
#include "bar.h"
export module n;
export using ::foo;
