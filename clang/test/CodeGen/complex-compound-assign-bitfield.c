// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -std=c23 -emit-llvm -o - | FileCheck %s

struct Bits {
    int pad1: 30;
    bool b: 1;
    int pad2: 1;
};

int main(void) {
// CHECK-LABEL: define dso_local i32 @main() #0 {
    struct Bits x;
    x.b += __builtin_complex(-1.0f, 0.0f);
// CHECK: %bf.load = load i32, ptr %x, align 4
// CHECK-NEXT: %bf.lshr = lshr i32 %bf.load, 30
// CHECK-NEXT: %bf.clear = and i32 %bf.lshr, 1
// CHECK-NEXT: %bf.cast = trunc i32 %bf.clear to i1
// CHECK-NEXT: %conv = uitofp i1 %bf.cast to float

// CHECK: %0 = zext i1 %tobool1 to i32
// CHECK-NEXT: %bf.load2 = load i32, ptr %x, align 4
// CHECK-NEXT: %bf.shl = shl i32 %0, 30
// CHECK-NEXT: %bf.clear3 = and i32 %bf.load2, -1073741825
// CHECK-NEXT: %bf.set = or i32 %bf.clear3, %bf.shl
// CHECK-NEXT: store i32 %bf.set, ptr %x, align 4
}
