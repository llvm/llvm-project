// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/a.cppm \
// RUN:     -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/b.cppm \
// RUN:     -emit-module-interface -fprebuilt-module-path=%t -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/c.cppm \
// RUN:     -emit-module-interface -fprebuilt-module-path=%t -o %t/c.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/c.pcm -S \
// RUN:     -emit-llvm -disable-llvm-passes -o - | FileCheck %t/c.cppm
//
// Be sure that we keep the same behavior as if optization not enabled.
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -O3 %t/a.cppm \
// RUN:     -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -O3 %t/b.cppm \
// RUN:     -emit-module-interface -fprebuilt-module-path=%t -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -O3 %t/c.cppm \
// RUN:     -emit-module-interface -fprebuilt-module-path=%t -o %t/c.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -O3 %t/c.pcm \
// RUN:     -S -emit-llvm -disable-llvm-passes -o - | FileCheck %t/c.cppm

//--- a.cppm
export module a;
export int a() {
    return 43;
}

template <int C>
int implicitly_inlined_template_function() {
    return C;
}

inline int reachable_inlined_a() {
    return 45;
}

int reachable_notinlined_a() {
    return 46;
}

export inline int inlined_a() {
    return 44 + reachable_inlined_a() +
           reachable_notinlined_a() +
           implicitly_inlined_template_function<47>();
}

//--- b.cppm
export module b;
export import a;
export int b() {
    return 43 + a();
}
export inline int inlined_b() {
    return 44 + inlined_a() + a();;
}

//--- c.cppm
export module c;
export import b;
export int c() {
    return 43 + b() + a() + inlined_b() + inlined_a();
}

// CHECK: declare{{.*}}@_ZW1b1bv
// CHECK: declare{{.*}}@_ZW1a1av
// CHECK: define{{.*}}@_ZW1b9inlined_bv
// CHECK: define{{.*}}@_ZW1a9inlined_av
// CHECK: define{{.*}}@_ZW1a19reachable_inlined_av
// CHECK: declare{{.*}}@_ZW1a22reachable_notinlined_av
// CHECK: define{{.*}}@_ZW1a36implicitly_inlined_template_functionILi47EEiv
