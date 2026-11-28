// RUN: %clang_cc1 -triple x86_64-w64-mingw32 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-w64-mingw32 -fexperimental-new-constant-interpreter -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-w64-mingw32 -emit-llvm -o - %s | FileCheck %s

// __builtin_imgrel32 is the offset of a global from the image base, which the
// linker resolves. It is emitted as a difference from __ImageBase.

struct locator {
  unsigned signature;
  unsigned self;
  unsigned data;
  unsigned element;
};

int data;
int array[4];
void func(void) {}

// CHECK: @locator = {{.*}}constant %struct.locator { i32 1, i32 trunc (i64 sub (i64 ptrtoint (ptr @locator to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @data to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr getelementptr {{.*}}(i8, ptr @array, i64 8) to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }
// CHECK: @__ImageBase = external {{(dso_local )?}}constant i8
const struct locator locator = {
  1, __builtin_imgrel32(&locator), __builtin_imgrel32(&data),
  __builtin_imgrel32(&array[2]),
};

// CHECK: @func_rva = {{.*}}constant i32 trunc (i64 sub (i64 ptrtoint (ptr @func to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32)
const unsigned func_rva = __builtin_imgrel32(func);

// It also works outside a constant initializer.
// CHECK-LABEL: define {{.*}}i32 @func_offset(
// CHECK: ret i32 trunc (i64 sub (i64 ptrtoint (ptr @func_offset to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32)
unsigned func_offset(void) { return __builtin_imgrel32(func_offset); }

// CHECK-LABEL: define {{.*}}i32 @data_offset(
// CHECK: ret i32 trunc (i64 sub (i64 ptrtoint (ptr getelementptr {{.*}}(i8, ptr @array, i64 4) to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32)
unsigned data_offset(void) { return __builtin_imgrel32(&array[1]); }
