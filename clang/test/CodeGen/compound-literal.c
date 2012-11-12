// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm %s -o - | FileCheck %s

int* a = &(int){1};
struct s {int a, b, c;} * b = &(struct s) {1, 2, 3};
_Complex double * x = &(_Complex double){1.0f};
typedef int v4i32 __attribute((vector_size(16)));
v4i32 *y = &(v4i32){1,2,3,4};

void xxx() {
int* a = &(int){1};
struct s {int a, b, c;} * b = &(struct s) {1, 2, 3};
_Complex double * x = &(_Complex double){1.0f};
}

// CHECK: define void @f()
void f() {
  typedef struct S { int x,y; } S;
  // CHECK: [[S:%[a-zA-Z0-9.]+]] = alloca [[STRUCT:%[a-zA-Z0-9.]+]],
  struct S s;
  // CHECK-NEXT: [[COMPOUNDLIT:%[a-zA-Z0-9.]+]] = alloca [[STRUCT]]
  // CHECK-NEXT: [[CX:%[a-zA-Z0-9.]+]] = getelementptr inbounds [[STRUCT]]* [[COMPOUNDLIT]], i32 0, i32 0
  // CHECK-NEXT: [[SY:%[a-zA-Z0-9.]+]] = getelementptr inbounds [[STRUCT]]* [[S]], i32 0, i32 1
  // CHECK-NEXT: [[TMP:%[a-zA-Z0-9.]+]] = load i32* [[SY]]
  // CHECK-NEXT: store i32 [[TMP]], i32* [[CX]]
  // CHECK-NEXT: [[CY:%[a-zA-Z0-9.]+]] = getelementptr inbounds [[STRUCT]]* [[COMPOUNDLIT]], i32 0, i32 1
  // CHECK-NEXT: [[SX:%[a-zA-Z0-9.]+]] = getelementptr inbounds [[STRUCT]]* [[S]], i32 0, i32 0
  // CHECK-NEXT: [[TMP:%[a-zA-Z0-9.]+]] = load i32* [[SX]]
  // CHECK-NEXT: store i32 [[TMP]], i32* [[CY]]
  // CHECK-NEXT: [[SI8:%[a-zA-Z0-9.]+]] = bitcast [[STRUCT]]* [[S]] to i8*
  // CHECK-NEXT: [[COMPOUNDLITI8:%[a-zA-Z0-9.]+]] = bitcast [[STRUCT]]* [[COMPOUNDLIT]] to i8*
  // CHECK-NEXT: call void @llvm.memcpy{{.*}}(i8* [[SI8]], i8* [[COMPOUNDLITI8]]
  s = (S){s.y,s.x};
  // CHECK-NEXT: ret void
}
