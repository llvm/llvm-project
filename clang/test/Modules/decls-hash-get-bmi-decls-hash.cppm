// Test that -module-file-info can print the decls hash for thin BMI.
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-thin-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/a.v1.cppm -emit-thin-module-interface -o %t/a.v1.pcm
//
// RUN: %clang_cc1 -get-bmi-decls-hash %t/a.pcm > %t/a.pcm.hash
// RUN: %clang_cc1 -get-bmi-decls-hash %t/a.v1.pcm > %t/a.v1.pcm.hash
//
// RUN: diff %t/a.pcm.hash %t/a.v1.pcm.hash

//--- a.cppm
export module a;
export int v = 43;
export int a() {
    return 43;
}

unsigned int non_exported() {
    return v;
}

//--- a.v1.cppm
export module a;
export int v = 45;
export int a() {
    return 44;
}

unsigned int non_exported() {
    return v + 43;
}
