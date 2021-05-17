// RUN: %clang_cc1 -emit-llvm -triple x86_64 %s -o - | FileCheck %s --check-prefixes=CHECK,CUSED
// RUN: %clang_cc1 -emit-llvm -triple x86_64-apple-darwin %s -o - | FileCheck %s --check-prefixes=CHECK,USED

// USED:       @llvm.used =
// CUSED:      @llvm.compiler.used =
// CHECK-SAME:    @f0
// CHECK-SAME:    @f1.l0
// CHECK-SAME:    @g0
// CHECK-SAME:    @a0

int g0 __attribute__((used));

static void __attribute__((used)) f0(void) {
}

void f1() { 
  static int l0 __attribute__((used)) = 5225; 
}

__attribute__((used)) int a0;
void pr27535() { (void)a0; }
