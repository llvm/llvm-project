// RUN: %clang_cc1 -emit-llvm -triple=i386-pc-win32 -fms-extensions %s -o - | FileCheck %s

enum Enum { zero, one, two };

struct __declspec(dllexport) S {
  // In MS compatibility mode, this counts as a definition.
  // Since it is exported, it must be emitted even if it's unreferenced.
  static const short x = 42;

  // This works for enums too.
  static const Enum y = two;

  struct NonExported {
    // dllexport is not inherited by this nested class.
    // Since z is not referenced, it should not be emitted.
    static const int z = 42;
  };
};

// CHECK: @"?x@S@@2FB" = weak_odr dso_local dllexport constant i16 42, comdat, align 2
// CHECK: @"?y@S@@2W4Enum@@B" = weak_odr dso_local dllexport constant i32 2, comdat, align 4
// CHECK-NOT: NonExported
