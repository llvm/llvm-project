// RUN: %clang -mfloat-abi=soft -target arm-linux-gnu -emit-llvm -S -o - %s | FileCheck %s

struct Empty {};

struct Empty emptyvar;

void take_args(int a, ...) {
// CHECK: [[ALLOCA_VA_LIST:%[a-zA-Z0-9._]+]] = alloca %struct.__va_list, align 4
// CHECK: call void @llvm.va_start

  // It's conceivable that EMPTY_PTR may not actually be a valid pointer
  // (e.g. it's at the very bottom of the stack and the next page is
  // invalid). This doesn't matter provided it's never loaded (there's no
  // well-defined way to tell), but it becomes a problem if we do try to use it.
// CHECK-NOT: load %struct.Empty
  __builtin_va_list l;
  __builtin_va_start(l, a);
  emptyvar = __builtin_va_arg(l, struct Empty);
  __builtin_va_end(l);
}
