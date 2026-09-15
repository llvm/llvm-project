// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O1 %s -o - | FileCheck %s

struct foo {
  int x;
  int y;
};

void bar(struct foo *);

// CHECK-LABEL: define dso_local void @baz()
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[SLOT1:.*]])
// CHECK: call void @bar(ptr noundef nonnull %[[SLOT1]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[SLOT1]])
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[SLOT2:.*]])
// CHECK: call void @bar(ptr noundef nonnull %[[SLOT2]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[SLOT2]])
void baz(void) {
  {
    bar(&(struct foo){.x = 42, .y = 25});
  }
  {
    bar(&(struct foo){.x = 77, .y = 99});
  }
}

int side_effect_true(void);
int side_effect_false(void);

// Verify that initializers with side effects in conditional expressions
// are evaluated at the point of expression evaluation (short-circuited),
// rather than hoisted to block entry.
// CHECK-LABEL: define dso_local void @test_conditional(
// CHECK: cond.true:
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[CL1:.*]])
// CHECK: call i32 @side_effect_true()
// CHECK: cond.false:
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[CL2:.*]])
// CHECK: call i32 @side_effect_false()
// CHECK: cond.end:
// CHECK: call void @bar(
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[CL2]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[CL1]])
void test_conditional(int cond) {
  cond ? bar(&(struct foo){.x = side_effect_true()})
       : bar(&(struct foo){.x = side_effect_false()});
}

int side_effect(void);

// Verify that initializers in logical short-circuiting operators are only
// evaluated when the RHS is executed.
// CHECK-LABEL: define dso_local void @test_short_circuit(
// CHECK: br i1 %{{.*}}, label %{{.*}}, label %[[LAND_RHS:.*]]
// CHECK: [[LAND_RHS]]:
// CHECK: call i32 @side_effect()
void test_short_circuit(int flag) {
  if (flag && ((struct foo){.x = side_effect()}).x > 0)
    bar(0);
}

void side_effect1(void);
int side_effect2(void);
void side_effect3(void);

// Verify side effects are sequenced in evaluation order, not hoisted to block entry.
// CHECK-LABEL: define dso_local void @test_order(
// CHECK: call void @side_effect1()
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[CL:.*]])
// CHECK: call i32 @side_effect2()
// CHECK: call void @bar(ptr noundef nonnull %[[CL]])
// CHECK: call void @side_effect3()
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[CL]])
void test_order(void) {
  side_effect1();
  bar(&(struct foo){.x = side_effect2()});
  side_effect3();
}

// Verify that lifetime markers are not emitted when a label has been seen in the
// current scope, matching VarDecl behavior to avoid miscompilation on backward jumps.
// CHECK-LABEL: define dso_local i32 @test_backward_goto()
// CHECK-NOT: call void @llvm.lifetime.start
// CHECK-NOT: call void @llvm.lifetime.end
// CHECK: ret i32
int test_backward_goto(void) {
  int *p = 0;
label:
  if (p)
    return *p;
  p = &(int){10};
  goto label;
}

