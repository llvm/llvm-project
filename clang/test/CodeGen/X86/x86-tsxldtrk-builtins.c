// RUN: %clang_cc1 %s -ffreestanding -triple i386-unknown-unknown -target-feature +tsxldtrk -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -ffreestanding -triple x86_64-unknown-unknown -target-feature +tsxldtrk -emit-llvm -o - | FileCheck %s

#include <immintrin.h>

void test_xsusldtrk() {
// CHECK-LABEL: test_xsusldtrk
// CHECK: call void @llvm.x86.xsusldtrk()
    _xsusldtrk();
}

void test_xresldtrk() {
// CHECK-LABEL: test_xresldtrk
// CHECK: call void @llvm.x86.xresldtrk()
    _xresldtrk();
}
