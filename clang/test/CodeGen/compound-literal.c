// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm %s -o - | FileCheck %s

// Capture the type and name so matching later is cleaner.
struct CompoundTy { int a; };
// CHECK: @MyCLH ={{.*}} constant [[MY_CLH:[^,]+]]
const struct CompoundTy *const MyCLH = &(struct CompoundTy){3};

int* a = &(int){1};
struct s {int a, b, c;} * b = &(struct s) {1, 2, 3};
_Complex double * x = &(_Complex double){1.0f};
typedef int v4i32 __attribute((vector_size(16)));
v4i32 *y = &(v4i32){1,2,3,4};

// Check generated code for GNU constant array init from compound literal,
// for a global variable.
// CHECK: @compound_array ={{.*}} global [8 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8]
int compound_array[] = __extension__(__builtin_choose_expr(0, 0, _Generic(1, int: (int[]){1, 2, 3, 4, 5, 6, 7, 8})));

void xxx() {
int* a = &(int){1};
struct s {int a, b, c;} * b = &(struct s) {1, 2, 3};
_Complex double * x = &(_Complex double){1.0f};
}

// CHECK-LABEL: define{{.*}} void @f()
void f() {
  typedef struct S { int x,y; } S;
  // CHECK: [[S:%[a-zA-Z0-9.]+]] = alloca [[STRUCT:%[a-zA-Z0-9.]+]],
  struct S s;
  // CHECK-NEXT: [[COMPOUNDLIT:%[a-zA-Z0-9.]+]] = alloca [[STRUCT]]
  // CHECK-NEXT: [[CX:%[a-zA-Z0-9.]+]] = getelementptr inbounds [[STRUCT]], [[STRUCT]]* [[COMPOUNDLIT]], i32 0, i32 0
  // CHECK-NEXT: [[SY:%[a-zA-Z0-9.]+]] = getelementptr inbounds [[STRUCT]], [[STRUCT]]* [[S]], i32 0, i32 1
  // CHECK-NEXT: [[TMP:%[a-zA-Z0-9.]+]] = load i32, i32* [[SY]]
  // CHECK-NEXT: store i32 [[TMP]], i32* [[CX]]
  // CHECK-NEXT: [[CY:%[a-zA-Z0-9.]+]] = getelementptr inbounds [[STRUCT]], [[STRUCT]]* [[COMPOUNDLIT]], i32 0, i32 1
  // CHECK-NEXT: [[SX:%[a-zA-Z0-9.]+]] = getelementptr inbounds [[STRUCT]], [[STRUCT]]* [[S]], i32 0, i32 0
  // CHECK-NEXT: [[TMP:%[a-zA-Z0-9.]+]] = load i32, i32* [[SX]]
  // CHECK-NEXT: store i32 [[TMP]], i32* [[CY]]
  // CHECK-NEXT: [[SI8:%[a-zA-Z0-9.]+]] = bitcast [[STRUCT]]* [[S]] to i8*
  // CHECK-NEXT: [[COMPOUNDLITI8:%[a-zA-Z0-9.]+]] = bitcast [[STRUCT]]* [[COMPOUNDLIT]] to i8*
  // CHECK-NEXT: call void @llvm.memcpy{{.*}}(i8* align {{[0-9]+}} [[SI8]], i8* align {{[0-9]+}} [[COMPOUNDLITI8]]
  s = (S){s.y,s.x};
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} i48 @g(
struct G { short x, y, z; };
struct G g(int x, int y, int z) {
  // CHECK:      [[RESULT:%.*]] = alloca [[G:%.*]], align 2
  // CHECK-NEXT: [[X:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[Y:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[Z:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[COERCE_TEMP:%.*]] = alloca i48
  // CHECK-NEXT: store i32
  // CHECK-NEXT: store i32
  // CHECK-NEXT: store i32

  // Evaluate the compound literal directly in the result value slot.
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[G]], [[G]]* [[RESULT]], i32 0, i32 0
  // CHECK-NEXT: [[T1:%.*]] = load i32, i32* [[X]], align 4
  // CHECK-NEXT: [[T2:%.*]] = trunc i32 [[T1]] to i16
  // CHECK-NEXT: store i16 [[T2]], i16* [[T0]], align 2
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[G]], [[G]]* [[RESULT]], i32 0, i32 1
  // CHECK-NEXT: [[T1:%.*]] = load i32, i32* [[Y]], align 4
  // CHECK-NEXT: [[T2:%.*]] = trunc i32 [[T1]] to i16
  // CHECK-NEXT: store i16 [[T2]], i16* [[T0]], align 2
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[G]], [[G]]* [[RESULT]], i32 0, i32 2
  // CHECK-NEXT: [[T1:%.*]] = load i32, i32* [[Z]], align 4
  // CHECK-NEXT: [[T2:%.*]] = trunc i32 [[T1]] to i16
  // CHECK-NEXT: store i16 [[T2]], i16* [[T0]], align 2
  return (struct G) { x, y, z };

  // CHECK-NEXT: [[T0:%.*]] = bitcast i48* [[COERCE_TEMP]] to i8*
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[G]]* [[RESULT]] to i8*
  // CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align {{[0-9]+}} [[T0]], i8* align {{[0-9]+}} [[T1]], i64 6
  // CHECK-NEXT: [[T0:%.*]] = load i48, i48* [[COERCE_TEMP]]
  // CHECK-NEXT: ret i48 [[T0]]
}

// We had a bug where we'd emit a new GlobalVariable for each time we used a
// const pointer to a variable initialized by a compound literal.
// CHECK-LABEL: define{{.*}} i32 @compareMyCLH() #0
int compareMyCLH() {
  // CHECK: store i8* bitcast ([[MY_CLH]] to i8*)
  const void *a = MyCLH;
  // CHECK: store i8* bitcast ([[MY_CLH]] to i8*)
  const void *b = MyCLH;
  return a == b;
}

// Check generated code for GNU constant array init from compound literal,
// for a local variable.
// CHECK-LABEL: define{{.*}} i32 @compound_array_fn()
// CHECK: [[COMPOUND_ARRAY:%.*]] = alloca [8 x i32]
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64({{.*}}, i64 32, i1 false)
int compound_array_fn() {
  int compound_array[] = (int[]){1,2,3,4,5,6,7,8};
  return compound_array[0];
}
