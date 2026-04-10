// Check that sum_offsets1 and sum_offsets2 are identical
// with and without -fms-kernel
// RUN: %clang_cc1 -fms-kernel  -fms-extensions -O2 -triple x86_64-windows-msvc -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fms-extensions -O2 -triple x86_64-windows-msvc -emit-llvm %s -o - | FileCheck %s

// CHECK:      define dso_local noundef i64 @"?sum_offsets1@@YA_KXZ"()
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret i64 632

// CHECK:      define dso_local noundef i64 @"?sum_offsets2@@YA_KXZ"()
// CHECK-NEXT: entry:
// CHECK-NEXT    ret i64 632

typedef unsigned long long ULONG_PTR;

#define MY_OFFSETOF(type, field) (ULONG_PTR)(&((type *)0)->field)

namespace foo {
typedef struct MyStruct {
  char b;
  struct X {
   long m0[10], m1;
  } x[10];
  long c;
} MyStruct;
}

__declspec(noinline) ULONG_PTR sum_offsets1() {
  return MY_OFFSETOF(foo::MyStruct, x[1].m0[2]) +
	 MY_OFFSETOF(foo::MyStruct, x[2].m1) +
	 MY_OFFSETOF(foo::MyStruct, c);
}

__declspec(noinline) ULONG_PTR sum_offsets2() {
  return __builtin_offsetof(foo::MyStruct, x[1].m0[2]) +
	 __builtin_offsetof(foo::MyStruct, x[2].m1) +
	 __builtin_offsetof(foo::MyStruct, c);
}

