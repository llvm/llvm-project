// REQUIRES: x86-registered-target

// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: %clang_cc1 -std=c++20 %s -pic-level 2 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++20 %s -pic-level 2 -fmodule-output=%t/m.pcm -emit-llvm -o - \
// RUN:     | FileCheck %s
//
// RUN: %clang_cc1 -std=c++20 %s -emit-module-interface -o %t/m.pcm
// RUN: %clang_cc1 -std=c++20 %t/m.pcm -pic-level 2 -emit-llvm -o - | FileCheck %s 
// RUN: %clang_cc1 -std=c++20 %t/m.pcm -emit-llvm -o - | FileCheck %s --check-prefix=NOPIC

export module m;
export int x;
export int func() {
    return x;
}

// CHECK: ![[METADATA_NUM:[0-9]+]] = !{{{.*}}, !"PIC Level", i32 2}
// NOPIC-NOT: ![[METADATA_NUM:[0-9]+]] = !{{{.*}}, !"PIC Level", i32 2}
