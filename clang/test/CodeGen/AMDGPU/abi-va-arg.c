// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -emit-llvm %s -o - | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -emit-llvm %s -o - | FileCheck %s %}

// Checks how AMDGPU functions read variadic arguments with va_arg.
// Each value is read in place from 4-byte slots of one buffer. The cursor
// moves by the alloc size rounded up to 4 and is never realigned.

typedef struct { char c; } char_t;
typedef struct { int a, b, c; } triple_t;
typedef struct { int data[20]; } big_t;

int va_int(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  int v = __builtin_va_arg(ap, int);
  __builtin_va_end(ap);
  return v;
}

// CHECK-LABEL: define {{.*}} i32 @va_int(i32 noundef %{{.*}}, ...)
// CHECK:         call void @llvm.va_start.p0(ptr %[[AP:[^)]+]])
// CHECK-NEXT:    %[[CUR:.*]] = load ptr, ptr %[[AP]], align 8
// CHECK-NEXT:    %[[NEXT:.*]] = getelementptr {{.*}}i8, ptr %[[CUR]], i64 4
// CHECK-NEXT:    store ptr %[[NEXT]], ptr %[[AP]], align 8
// CHECK-NEXT:    %{{.*}} = load i32, ptr %[[CUR]], align 4

// A double takes two slots and is read at the slot's alignment, not its own 8.
double va_double(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  double v = __builtin_va_arg(ap, double);
  __builtin_va_end(ap);
  return v;
}

// CHECK-LABEL: define {{.*}} double @va_double(i32 noundef %{{.*}}, ...)
// CHECK:         call void @llvm.va_start.p0(ptr %[[AP:[^)]+]])
// CHECK-NEXT:    %[[CUR:.*]] = load ptr, ptr %[[AP]], align 8
// CHECK-NEXT:    %[[NEXT:.*]] = getelementptr {{.*}}i8, ptr %[[CUR]], i64 8
// CHECK-NEXT:    store ptr %[[NEXT]], ptr %[[AP]], align 8
// CHECK-NEXT:    %{{.*}} = load double, ptr %[[CUR]], align 4

// A one-byte record still takes a whole slot.
char_t va_char_struct(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  char_t v = __builtin_va_arg(ap, char_t);
  __builtin_va_end(ap);
  return v;
}

// CHECK-LABEL: define {{.*}} i8 @va_char_struct(i32 noundef %{{.*}}, ...)
// CHECK:         call void @llvm.va_start.p0(ptr %[[AP:[^)]+]])
// CHECK-NEXT:    %[[CUR:.*]] = load ptr, ptr %[[AP]], align 8
// CHECK-NEXT:    %[[NEXT:.*]] = getelementptr {{.*}}i8, ptr %[[CUR]], i64 4
// CHECK-NEXT:    store ptr %[[NEXT]], ptr %[[AP]], align 8
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 1 %{{.*}}, ptr align 4 %[[CUR]], i64 1, i1 false)

triple_t va_triple(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  triple_t v = __builtin_va_arg(ap, triple_t);
  __builtin_va_end(ap);
  return v;
}

// CHECK-LABEL: define {{.*}} %struct.triple_t @va_triple(i32 noundef %{{.*}}, ...)
// CHECK:         call void @llvm.va_start.p0(ptr %[[AP:[^)]+]])
// CHECK-NEXT:    %[[CUR:.*]] = load ptr, ptr %[[AP]], align 8
// CHECK-NEXT:    %[[NEXT:.*]] = getelementptr {{.*}}i8, ptr %[[CUR]], i64 12
// CHECK-NEXT:    store ptr %[[NEXT]], ptr %[[AP]], align 8
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %[[CUR]], i64 12, i1 false)

// A record past the register budget is still read in place, not through a
// pointer.
big_t va_big(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  big_t v = __builtin_va_arg(ap, big_t);
  __builtin_va_end(ap);
  return v;
}

// CHECK-LABEL: define {{.*}} void @va_big(ptr addrspace(5) dead_on_unwind noalias writable sret(%struct.big_t) align 4 %{{.*}}, i32 noundef %{{.*}}, ...)
// CHECK:         call void @llvm.va_start.p0(ptr %[[AP:[^)]+]])
// CHECK-NEXT:    %[[CUR:.*]] = load ptr, ptr %[[AP]], align 8
// CHECK-NEXT:    %[[NEXT:.*]] = getelementptr {{.*}}i8, ptr %[[CUR]], i64 80
// CHECK-NEXT:    store ptr %[[NEXT]], ptr %[[AP]], align 8
// CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %[[CUR]], i64 80, i1 false)
