// RUN: %clang_cc1 -no-opaque-pointers -triple arm64-apple-ios -target-abi darwinpcs -emit-llvm -o - %s | FileCheck %s
struct Empty {};

Empty emptyvar;

int take_args(int a, ...) {
  __builtin_va_list l;
  __builtin_va_start(l, a);
// CHECK: call void @llvm.va_start

  emptyvar = __builtin_va_arg(l, Empty);
// CHECK: load i8*, i8**
// CHECK-NOT: getelementptr
// CHECK: [[EMPTY_PTR:%[a-zA-Z0-9._]+]] = bitcast i8* {{%[a-zA-Z0-9._]+}} to %struct.Empty*

  // It's conceivable that EMPTY_PTR may not actually be a valid pointer
  // (e.g. it's at the very bottom of the stack and the next page is
  // invalid). This doesn't matter provided it's never loaded (there's no
  // well-defined way to tell), but it becomes a problem if we do try to use it.
// CHECK-NOT: load %struct.Empty, %struct.Empty* [[EMPTY_PTR]]

  int i = __builtin_va_arg(l, int);
// CHECK: va_arg i8** {{%[a-zA-Z0-9._]+}}, i32

  __builtin_va_end(l);
  return i;
}
