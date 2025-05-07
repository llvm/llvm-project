

// RUN: %clang_cc1 %s -O0 -fbounds-safety -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -O2 -fbounds-safety -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -O0 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -O2 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm -o - | FileCheck %s

#include <ptrcheck.h>

int f1(char *__bidi_indexable arr, char i) {
    return arr[i];
    // CHECK: {{%.*}} = sext i8 {{%.*}} to i64
}

int f2(char *__bidi_indexable arr, unsigned char i) {
    return arr[i];
    // CHECK: {{%.*}} = zext i8 {{%.*}} to i64
}
