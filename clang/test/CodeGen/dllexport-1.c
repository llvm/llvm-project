// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -fms-extensions -Wno-ignored-attributes -Wno-extern-initializer -o - %s | FileCheck %s -check-prefix CHECK-LNX
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm -fms-extensions -o - -DMSVC %s | FileCheck %s -check-prefix CHECK-MSVC

// Export const variable.

// CHECK-MSVC: @x = dso_local dllexport constant i32 3, align 4
// CHECK-LNX: @x ={{.*}} constant i32 3, align 4

// CHECK-MSVC: @z = dso_local constant i32 4, align 4
// CHECK-LNX: @z ={{.*}} constant i32 4, align 4

// CHECK-MSVC: @y = dso_local dllexport constant i32 0, align 4
// CHECK-LNX: @y ={{.*}} constant i32 0, align 4

__declspec(dllexport) int const x = 3;
__declspec(dllexport) const int y;

// expected-warning@+1 {{'extern' variable has an initializer}}
extern int const z = 4;

int main(void) {
  int a = x + y + z;
  return a;
}
