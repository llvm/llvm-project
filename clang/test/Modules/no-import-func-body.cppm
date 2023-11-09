// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 -O1 -triple %itanium_abi_triple %t/a.cppm \
// RUN:     -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -O1 -triple %itanium_abi_triple %t/b.cppm \
// RUN:     -emit-module-interface -fprebuilt-module-path=%t -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 -O1 -triple %itanium_abi_triple %t/c.cppm \
// RUN:     -emit-module-interface -fprebuilt-module-path=%t -o %t/c.pcm
// RUN: %clang_cc1 -std=c++20 -O1 -triple %itanium_abi_triple %t/c.pcm -S \
// RUN:     -emit-llvm -disable-llvm-passes -o - | FileCheck %t/c.cppm

//--- a.cppm
export module a;
export int a() {
    return 43;
}
export __attribute__((noinline)) int a_noinline() {
    return 44;
}

//--- b.cppm
export module b;
export import a;
export int b() {
    return 43 + a();
}

export __attribute__((noinline)) int b_noinline() {
    return 43 + a();
}

//--- c.cppm
export module c;
export import b;
export int c() {
    return 43 + b() + a() + b_noinline() + a_noinline();
}

// CHECK: declare{{.*}}@_ZW1b1bv(
// CHECK: declare{{.*}}@_ZW1a1av(

// CHECK: declare{{.*}}@_ZW1b10b_noinlinev()
// CHECK: declare{{.*}}@_ZW1a10a_noinlinev()
