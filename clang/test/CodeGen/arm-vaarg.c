// RUN: %clang -Xclang -no-opaque-pointers -mfloat-abi=soft -target arm-linux-gnu -emit-llvm -S -o - %s | FileCheck %s

struct Empty {};

struct Empty emptyvar;

void take_args(int a, ...) {
// CHECK: [[ALLOCA_VA_LIST:%[a-zA-Z0-9._]+]] = alloca %struct.__va_list, align 4
// CHECK: call void @llvm.va_start
// CHECK-NEXT: [[AP_ADDR:%[a-zA-Z0-9._]+]] = bitcast %struct.__va_list* [[ALLOCA_VA_LIST]] to i8**
// CHECK-NEXT: [[LOAD_AP:%[a-zA-Z0-9._]+]] = load i8*, i8** [[AP_ADDR]], align 4
// CHECK-NEXT: [[EMPTY_PTR:%[a-zA-Z0-9._]+]] = bitcast i8* [[LOAD_AP]] to %struct.Empty*

  // It's conceivable that EMPTY_PTR may not actually be a valid pointer
  // (e.g. it's at the very bottom of the stack and the next page is
  // invalid). This doesn't matter provided it's never loaded (there's no
  // well-defined way to tell), but it becomes a problem if we do try to use it.
// CHECK-NOT: load %struct.Empty, %struct.Empty* [[EMPTY_PTR]]
  __builtin_va_list l;
  __builtin_va_start(l, a);
  emptyvar = __builtin_va_arg(l, struct Empty);
  __builtin_va_end(l);
}
