// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o - -fblocks | FileCheck %s

// CHECK: %[[STRUCT_BLOCK_DESCRIPTOR:.*]] = type { i32, i32 }

// CHECK: @{{.*}} = internal constant { i32, i32, i8*, i8*, i8*, i8* } { i32 0, i32 24, i8* bitcast (void (i8*, i8*)* @__copy_helper_block_4_20r to i8*), i8* bitcast (void (i8*)* @__destroy_helper_block_4_20r to i8*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @{{.*}}, i32 0, i32 0), i8* null }, align 4
// CHECK: @[[BLOCK_DESCRIPTOR_TMP21:.*]] = internal constant { i32, i32, i8*, i8*, i8*, i8* } { i32 0, i32 24, i8* bitcast (void (i8*, i8*)* @__copy_helper_block_4_20r to i8*), i8* bitcast (void (i8*)* @__destroy_helper_block_4_20r to i8*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @{{.*}}, i32 0, i32 0), i8* null }, align 4

void (^f)(void) = ^{};

// rdar://6768379
int f0(int (^a0)()) {
  return a0(1, 2, 3);
}

// Verify that attributes on blocks are set correctly.
typedef struct s0 T;
struct s0 {
  int a[64];
};

// CHECK: define internal void @__f2_block_invoke(%struct.s0* noalias sret {{%.*}}, i8* {{%.*}}, %struct.s0* byval(%struct.s0) align 4 {{.*}})
struct s0 f2(struct s0 a0) {
  return ^(struct s0 a1){ return a1; }(a0);
}

// This should not crash: rdar://6808051
void *P = ^{
  void *Q = __func__;
};

void (^test1)(void) = ^(void) {
  __block int i;
  ^ { i = 1; }();
};

// CHECK-LABEL: define linkonce_odr hidden void @__copy_helper_block_4_20r(i8*, i8*) unnamed_addr
// CHECK: %[[_ADDR:.*]] = alloca i8*, align 4
// CHECK-NEXT: %[[_ADDR1:.*]] = alloca i8*, align 4
// CHECK-NEXT: store i8* %0, i8** %[[_ADDR]], align 4
// CHECK-NEXT: store i8* %1, i8** %[[_ADDR1]], align 4
// CHECK-NEXT: %[[V2:.*]] = load i8*, i8** %[[_ADDR1]], align 4
// CHECK-NEXT: %[[BLOCK_SOURCE:.*]] = bitcast i8* %[[V2]] to <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>*
// CHECK-NEXT: %[[V3:.*]] = load i8*, i8** %[[_ADDR]], align 4
// CHECK-NEXT: %[[BLOCK_DEST:.*]] = bitcast i8* %[[V3]] to <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>*
// CHECK-NEXT: %[[V4:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>* %[[BLOCK_SOURCE]], i32 0, i32 5
// CHECK-NEXT: %[[V5:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>* %[[BLOCK_DEST]], i32 0, i32 5
// CHECK-NEXT: %[[BLOCKCOPY_SRC:.*]] = load i8*, i8** %[[V4]], align 4
// CHECK-NEXT: %[[V6:.*]] = bitcast i8** %[[V5]] to i8*
// CHECK-NEXT: call void @_Block_object_assign(i8* %[[V6]], i8* %[[BLOCKCOPY_SRC]], i32 8)
// CHECK-NEXT: ret void

// CHECK-LABEL: define linkonce_odr hidden void @__destroy_helper_block_4_20r(i8*) unnamed_addr
// CHECK: %[[_ADDR:.*]] = alloca i8*, align 4
// CHECK-NEXT: store i8* %0, i8** %[[_ADDR]], align 4
// CHECK-NEXT: %[[V1:.*]] = load i8*, i8** %[[_ADDR]], align 4
// CHECK-NEXT: %[[BLOCK:.*]] = bitcast i8* %[[V1]] to <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>*
// CHECK-NEXT: %[[V2:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>* %[[BLOCK]], i32 0, i32 5
// CHECK-NEXT: %[[V3:.*]] = load i8*, i8** %[[V2]], align 4
// CHECK-NEXT: call void @_Block_object_dispose(i8* %[[V3]], i32 8)
// CHECK-NEXT: ret void

typedef double ftype(double);
// It's not clear that we *should* support this syntax, but until that decision
// is made, we should support it properly and not crash.
ftype ^test2 = ^ftype {
  return 0;
};

// rdar://problem/8605032
void f3_helper(void (^)(void));
void f3() {
  _Bool b = 0;
  f3_helper(^{ if (b) {} });
}

// rdar://problem/11322251
// The bool can fill in between the header and the long long.
// Add the appropriate amount of padding between them.
void f4_helper(long long (^)(void));
// CHECK-LABEL: define void @f4()
void f4(void) {
  _Bool b = 0;
  long long ll = 0;
  // CHECK: alloca <{ i8*, i32, i32, i8*, {{%.*}}*, i8, [3 x i8], i64 }>, align 8
  f4_helper(^{ if (b) return ll; return 0LL; });
}

// rdar://problem/11354538
// The alignment after rounding up to the align of F5 is actually
// greater than the required alignment.  Don't assert.
struct F5 {
  char buffer[32] __attribute((aligned));
};
void f5_helper(void (^)(struct F5 *));
// CHECK-LABEL: define void @f5()
void f5(void) {
  struct F5 value;
  // CHECK: alloca <{ i8*, i32, i32, i8*, {{%.*}}*, [12 x i8], [[F5:%.*]] }>, align 16
  f5_helper(^(struct F5 *slot) { *slot = value; });
}

// rdar://14085217
void (^b)() = ^{};
int main() {
   (b?: ^{})();
}
// CHECK: [[ZERO:%.*]] = load void (...)*, void (...)** @b
// CHECK-NEXT: [[TB:%.*]] = icmp ne void (...)* [[ZERO]], null
// CHECK-NEXT: br i1 [[TB]], label [[CT:%.*]], label [[CF:%.*]]
// CHECK: [[ONE:%.*]] = bitcast void (...)* [[ZERO]] to void ()*
// CHECK-NEXT:   br label [[CE:%.*]]

// Ensure that we don't emit helper code in copy/dispose routines for variables
// that are const-captured.
void testConstCaptureInCopyAndDestroyHelpers() {
  const int x = 0;
  __block int i;
  (^ { i = x; })();
}
// CHECK-LABEL: define void @testConstCaptureInCopyAndDestroyHelpers(
// CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>* %{{.*}}, i32 0, i32 4
// CHECK: store %[[STRUCT_BLOCK_DESCRIPTOR]]* bitcast ({ i32, i32, i8*, i8*, i8*, i8* }* @[[BLOCK_DESCRIPTOR_TMP21]] to %[[STRUCT_BLOCK_DESCRIPTOR]]*), %[[STRUCT_BLOCK_DESCRIPTOR]]** %[[BLOCK_DESCRIPTOR]], align 4

// CHECK-LABEL: define internal void @__testConstCaptureInCopyAndDestroyHelpers_block_invoke
