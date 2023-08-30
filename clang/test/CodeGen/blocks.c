// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -Wno-strict-prototypes -o - -fblocks | FileCheck %s

// CHECK: @{{.*}} = internal constant { i32, i32, ptr, ptr, ptr, ptr } { i32 0, i32 24, ptr @__copy_helper_block_4_20r, ptr @__destroy_helper_block_4_20r, ptr @{{.*}}, ptr null }, align 4
// CHECK: @[[BLOCK_DESCRIPTOR_TMP21:.*]] = internal constant { i32, i32, ptr, ptr, ptr, ptr } { i32 0, i32 24, ptr @__copy_helper_block_4_20r, ptr @__destroy_helper_block_4_20r, ptr @{{.*}}, ptr null }, align 4

void (^f)(void) = ^{};

int f0(int (^a0)()) {
  return a0(1, 2, 3);
}

// Verify that attributes on blocks are set correctly.
typedef struct s0 T;
struct s0 {
  int a[64];
};

// CHECK: define internal void @__f2_block_invoke(ptr noalias sret(%struct.s0) align 4 {{%.*}}, ptr noundef {{%.*}}, ptr noundef byval(%struct.s0) align 4 {{.*}})
struct s0 f2(struct s0 a0) {
  return ^(struct s0 a1){ return a1; }(a0);
}

// This should not crash.
void *P = ^{
  void *Q = __func__;
};

void (^test1)(void) = ^(void) {
  __block int i;
  ^ { i = 1; }();
};

// CHECK-LABEL: define linkonce_odr hidden void @__copy_helper_block_4_20r(ptr noundef %0, ptr noundef %1) unnamed_addr
// CHECK: %[[_ADDR:.*]] = alloca ptr, align 4
// CHECK-NEXT: %[[_ADDR1:.*]] = alloca ptr, align 4
// CHECK-NEXT: store ptr %0, ptr %[[_ADDR]], align 4
// CHECK-NEXT: store ptr %1, ptr %[[_ADDR1]], align 4
// CHECK-NEXT: %[[V2:.*]] = load ptr, ptr %[[_ADDR1]], align 4
// CHECK-NEXT: %[[V3:.*]] = load ptr, ptr %[[_ADDR]], align 4
// CHECK-NEXT: %[[V4:.*]] = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %[[V2]], i32 0, i32 5
// CHECK-NEXT: %[[V5:.*]] = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %[[V3]], i32 0, i32 5
// CHECK-NEXT: %[[BLOCKCOPY_SRC:.*]] = load ptr, ptr %[[V4]], align 4
// CHECK-NEXT: call void @_Block_object_assign(ptr %[[V5]], ptr %[[BLOCKCOPY_SRC]], i32 8)
// CHECK-NEXT: ret void

// CHECK-LABEL: define linkonce_odr hidden void @__destroy_helper_block_4_20r(ptr noundef %0) unnamed_addr
// CHECK: %[[_ADDR:.*]] = alloca ptr, align 4
// CHECK-NEXT: store ptr %0, ptr %[[_ADDR]], align 4
// CHECK-NEXT: %[[V1:.*]] = load ptr, ptr %[[_ADDR]], align 4
// CHECK-NEXT: %[[V2:.*]] = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %[[V1]], i32 0, i32 5
// CHECK-NEXT: %[[V3:.*]] = load ptr, ptr %[[V2]], align 4
// CHECK-NEXT: call void @_Block_object_dispose(ptr %[[V3]], i32 8)
// CHECK-NEXT: ret void

typedef double ftype(double);
// It's not clear that we *should* support this syntax, but until that decision
// is made, we should support it properly and not crash.
ftype ^test2 = ^ftype {
  return 0;
};

void f3_helper(void (^)(void));
void f3(void) {
  _Bool b = 0;
  f3_helper(^{ if (b) {} });
}

// The bool can fill in between the header and the long long.
// Add the appropriate amount of padding between them.
void f4_helper(long long (^)(void));
// CHECK-LABEL: define{{.*}} void @f4()
void f4(void) {
  _Bool b = 0;
  long long ll = 0;
  // CHECK: alloca <{ ptr, i32, i32, ptr, ptr, i8, [3 x i8], i64 }>, align 8
  f4_helper(^{ if (b) return ll; return 0LL; });
}

// The alignment after rounding up to the align of F5 is actually
// greater than the required alignment.  Don't assert.
struct F5 {
  char buffer[32] __attribute((aligned));
};
void f5_helper(void (^)(struct F5 *));
// CHECK-LABEL: define{{.*}} void @f5()
void f5(void) {
  struct F5 value;
  // CHECK: alloca <{ ptr, i32, i32, ptr, ptr, [12 x i8], [[F5:%.*]] }>, align 16
  f5_helper(^(struct F5 *slot) { *slot = value; });
}

void (^b)() = ^{};
int main(void) {
   (b?: ^{})();
}
// CHECK: [[ZERO:%.*]] = load ptr, ptr @b
// CHECK-NEXT: [[TB:%.*]] = icmp ne ptr [[ZERO]], null
// CHECK-NEXT: br i1 [[TB]], label [[CT:%.*]], label [[CF:%.*]]
// CHECK:   br label [[CE:%.*]]

// Ensure that we don't emit helper code in copy/dispose routines for variables
// that are const-captured.
void testConstCaptureInCopyAndDestroyHelpers(void) {
  const int x = 0;
  __block int i;
  (^ { i = x; })();
}
// CHECK-LABEL: define{{.*}} void @testConstCaptureInCopyAndDestroyHelpers(
// CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %{{.*}}, i32 0, i32 4
// CHECK: store ptr @[[BLOCK_DESCRIPTOR_TMP21]], ptr %[[BLOCK_DESCRIPTOR]], align 4

// CHECK-LABEL: define internal void @__testConstCaptureInCopyAndDestroyHelpers_block_invoke
