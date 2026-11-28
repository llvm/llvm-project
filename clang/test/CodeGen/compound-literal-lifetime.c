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

// Verify that tight full-expression lifetime markers are not emitted when a
// compound literal is crossed by a backward jump in its enclosing block.
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

// Verify that a backward jump confined to an earlier statement expression does
// not suppress lifetime markers for subsequent compound literals in the outer
// block.
// CHECK-LABEL: define dso_local void @test_stmt_expr_label(
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[CL_SE:.*]])
// CHECK: call void @bar(ptr noundef nonnull %[[CL_SE]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[CL_SE]])
void test_stmt_expr_label(int n) {
  ({
    __label__ l;
  l:
    if (n--)
      goto l;
  });
  bar(&(struct foo){.x = 1, .y = 2});
}

// Verify that a label targeted only by a forward jump does not suppress
// lifetime markers for compound literals after the label.
// CHECK-LABEL: define dso_local void @test_label_without_backward_goto(
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[CL_FWD:.*]])
// CHECK: call void @bar(ptr noundef nonnull %[[CL_FWD]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[CL_FWD]])
void test_label_without_backward_goto(int cond) {
  if (cond)
    goto label;
  side_effect1();
label:
  bar(&(struct foo){.x = 1, .y = 2});
}

// Verify that jumping forward over a compound literal does not suppress its
// lifetime markers since anonymous compound literals cannot be referenced at
// the jump target.
// CHECK-LABEL: define dso_local void @test_forward_goto_bypass(
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[CL_FWD:.*]])
// CHECK: call void @bar(ptr noundef nonnull %[[CL_FWD]])
// CHECK: call void @side_effect1()
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[CL_FWD]])
void test_forward_goto_bypass(int cond) {
  if (cond)
    goto after;
  bar(&(struct foo){.x = 1, .y = 2});
after:
  side_effect1();
}

// Verify that a compound literal inside an inner block keeps its tight
// lifetime markers when a backward goto branches to a label outside that inner
// block.
// CHECK-LABEL: define dso_local void @test_inner_block_backward_goto_to_outer(
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[CL_IN:.*]])
// CHECK: call void @bar(ptr noundef nonnull %[[CL_IN]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[CL_IN]])
void test_inner_block_backward_goto_to_outer(int cond) {
outer:
  {
    bar(&(struct foo){.x = 1, .y = 2});
    if (cond--)
      goto outer;
  }
}

int bar_int(struct foo *);

// Verify that a compound literal in a switch condition is evaluated before the
// switch jump and retains its lifetime markers.
// CHECK-LABEL: define dso_local void @test_switch_cond_compound_literal(
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[CL_SW:.*]])
// CHECK: call i32 @bar_int(ptr noundef nonnull %[[CL_SW]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[CL_SW]])
void test_switch_cond_compound_literal(void) {
  switch (bar_int(&(struct foo){.x = 1, .y = 2})) {
  case 0:
    side_effect1();
    break;
  default:
    break;
  }
}

int *id_ptr(int *);

// Verify that an attributed statement does not hide a compound literal in a
// synthetic scope when crossed by a backward jump.
// CHECK-LABEL: define dso_local i32 @test_attributed_stmt_backward_goto()
// CHECK-NOT: call void @llvm.lifetime.start
// CHECK-NOT: call void @llvm.lifetime.end
// CHECK: ret i32
int test_attributed_stmt_backward_goto(void) {
  int *p = 0;
label:
  if (p)
    return *p;
  [[clang::nomerge]] p = id_ptr(&(int){10});
  goto label;
}

void use_vla(int, char *);

// Verify that a compound literal inside a VLA bound is visited during bypass
// detection when crossed by a backward jump.
// CHECK-LABEL: define dso_local i32 @test_vla_bound_backward_goto()
// CHECK-NOT: call void @llvm.lifetime
// CHECK: call ptr @id_ptr(
// CHECK-NOT: call void @llvm.lifetime
int test_vla_bound_backward_goto(void) {
  int *p = 0;
label:
  if (p)
    return *p;
  char vla[*(p = id_ptr(&(int){5}))];
  use_vla(sizeof(vla), vla);
  goto label;
}

// Verify that a compound literal inside a VLA typedef crossed by a backward
// jump suppresses its lifetime markers.
// CHECK-LABEL: define dso_local i32 @test_vla_typedef_backward_goto()
// CHECK-NOT: call void @llvm.lifetime
// CHECK: call ptr @id_ptr(
// CHECK-NOT: call void @llvm.lifetime
int test_vla_typedef_backward_goto(void) {
  int *p = 0;
label:
  if (p)
    return *p;
  typedef char vla_t[*(p = id_ptr(&(int){5}))];
  vla_t a;
  use_vla(sizeof(a), a);
  goto label;
}

// Verify pointer-to-VLA variably modified type.
// CHECK-LABEL: define dso_local i32 @test_vla_ptr_backward_goto()
// CHECK-NOT: call void @llvm.lifetime
// CHECK: call ptr @id_ptr(
// CHECK-NOT: call void @llvm.lifetime
int test_vla_ptr_backward_goto(void) {
  int *p = 0;
label:
  if (p)
    return *p;
  int (*vp)[*(p = id_ptr(&(int){5}))] = 0;
  (void)vp;
  goto label;
}

// Verify _Atomic variably modified type (AtomicType).
// CHECK-LABEL: define dso_local i32 @test_atomic_vla_ptr_backward_goto()
// CHECK-NOT: call void @llvm.lifetime
// CHECK: call ptr @id_ptr(
// CHECK-NOT: call void @llvm.lifetime
int test_atomic_vla_ptr_backward_goto(void) {
  int *p = 0;
label:
  if (p)
    return *p;
  _Atomic(int (*)[*(p = id_ptr(&(int){5}))]) vp = 0;
  (void)vp;
  goto label;
}

// Verify __typeof__(expr) with variably modified type (TypeOfExprType).
// CHECK-LABEL: define dso_local i32 @test_typeof_expr_vla_backward_goto()
// CHECK-NOT: call void @llvm.lifetime
// CHECK: call ptr @id_ptr(
// CHECK-NOT: call void @llvm.lifetime
int test_typeof_expr_vla_backward_goto(void) {
  int n = 5;
  int *p = 0;
label:
  if (p)
    return *p;
  char a[n];
  __typeof__(p = id_ptr(&(int){5}), &a) bp = 0;
  (void)bp;
  goto label;
}

// Verify __typeof__(type) with variably modified type (TypeOfType).
// CHECK-LABEL: define dso_local i32 @test_typeof_type_vla_backward_goto()
// CHECK-NOT: call void @llvm.lifetime
// CHECK: call ptr @id_ptr(
// CHECK-NOT: call void @llvm.lifetime
int test_typeof_type_vla_backward_goto(void) {
  int *p = 0;
label:
  if (p)
    return *p;
  __typeof__(char[*(p = id_ptr(&(int){5}))]) a;
  use_vla(sizeof(a), a);
  goto label;
}

// Verify sizeof(type) with a VLA type argument.
// CHECK-LABEL: define dso_local i32 @test_sizeof_vla_type_backward_goto()
// CHECK-NOT: call void @llvm.lifetime
// CHECK: call ptr @id_ptr(
// CHECK-NOT: call void @llvm.lifetime
int test_sizeof_vla_type_backward_goto(void) {
  int *p = 0;
label:
  if (p)
    return *p;
  (void)sizeof(char[*(p = id_ptr(&(int){5}))]);
  goto label;
}

// Verify sizeof(expr) whose argument has a VLA type.
// CHECK-LABEL: define dso_local i32 @test_sizeof_vla_expr_backward_goto()
// CHECK-NOT: call void @llvm.lifetime
// CHECK: call ptr @id_ptr(
// CHECK-NOT: call void @llvm.lifetime
int test_sizeof_vla_expr_backward_goto(void) {
  int n = 5;
  int *p = 0;
label:
  if (p)
    return *p;
  char vla[n];
  (void)sizeof(*(p = id_ptr(&(int){5}), &vla));
  goto label;
}

// Verify explicit cast to a pointer-to-VLA type.
// CHECK-LABEL: define dso_local i32 @test_cast_vla_ptr_backward_goto()
// CHECK-NOT: call void @llvm.lifetime
// CHECK: call ptr @id_ptr(
// CHECK-NOT: call void @llvm.lifetime
int test_cast_vla_ptr_backward_goto(void) {
  int *p = 0;
label:
  if (p)
    return *p;
  (void)(int (*)[*(p = id_ptr(&(int){5}))])0;
  goto label;
}

// Verify multi-dimensional ConstantArray of VariableArray.
// CHECK-LABEL: define dso_local i32 @test_multidim_vla_backward_goto()
// CHECK-NOT: call void @llvm.lifetime
// CHECK: call ptr @id_ptr(
// CHECK-NOT: call void @llvm.lifetime
int test_multidim_vla_backward_goto(void) {
  int *p = 0;
label:
  if (p)
    return *p;
  char vla[2][*(p = id_ptr(&(int){5}))];
  use_vla(sizeof(vla), vla[0]);
  goto label;
}

// Verify function pointer returning pointer to VLA (FunctionProtoType + Paren).
// CHECK-LABEL: define dso_local i32 @test_fnptr_ret_vla_backward_goto()
// CHECK-NOT: call void @llvm.lifetime
// CHECK: call ptr @id_ptr(
// CHECK-NOT: call void @llvm.lifetime
int test_fnptr_ret_vla_backward_goto(void) {
  int *p = 0;
label:
  if (p)
    return *p;
  int (*(*fp)(void))[*(p = id_ptr(&(int){5}))] = 0;
  (void)fp;
  goto label;
}

// Verify compound literal whose own type is variably modified.
// CHECK-LABEL: define dso_local i32 @test_cl_vm_type_backward_goto()
// CHECK-NOT: call void @llvm.lifetime
// CHECK: call ptr @id_ptr(
// CHECK-NOT: call void @llvm.lifetime
int test_cl_vm_type_backward_goto(void) {
  int *p = 0;
label:
  if (p)
    return *p;
  (void)&(int (*)[*(p = id_ptr(&(int){5}))]){0};
  goto label;
}

// Verify non-bypassed compound literals in variably modified types emit
// lifetime markers cleanly without leaving unpopped cleanups.
// CHECK-LABEL: define dso_local void @test_vla_non_bypassed()
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[VLA_CL:.*]])
// CHECK: call ptr @id_ptr(ptr noundef nonnull %[[VLA_CL]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[VLA_CL]])
void test_vla_non_bypassed(void) {
  char vla[*(id_ptr(&(int){5}))];
  use_vla(sizeof(vla), vla);
}

// Verify a compound literal inside a VLA typedef before a backward-jump label
// retains its lifetime markers even when the typedef is used after the label.
// CHECK-LABEL: define dso_local i32 @test_vla_typedef_before_label()
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[TD_CL:.*]])
// CHECK: call ptr @id_ptr(ptr noundef nonnull %[[TD_CL]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[TD_CL]])
int test_vla_typedef_before_label(void) {
  int *p = 0;
  typedef char vla_t[*(p = id_ptr(&(int){5}))];
  int count = 0;
label:
  if (count++)
    return *p;
  vla_t a;
  use_vla(sizeof(a), a);
  goto label;
}

// CHECK-LABEL: define dso_local void @test_sizeof_vla_non_bypassed()
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[SZ_CL:.*]])
// CHECK: call ptr @id_ptr(ptr noundef nonnull %[[SZ_CL]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[SZ_CL]])
void test_sizeof_vla_non_bypassed(void) {
  (void)sizeof(char[*(id_ptr(&(int){5}))]);
}

// CHECK-LABEL: define dso_local void @test_cast_vla_non_bypassed()
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[CAST_CL:.*]])
// CHECK: call ptr @id_ptr(ptr noundef nonnull %[[CAST_CL]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[CAST_CL]])
void test_cast_vla_non_bypassed(void) {
  (void)(int (*)[*(id_ptr(&(int){5}))])0;
}

// CHECK-LABEL: define dso_local void @test_typeof_expr_uninit_non_bypassed()
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[TO_CL:.*]])
// CHECK: call ptr @id_ptr(ptr noundef nonnull %[[TO_CL]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[TO_CL]])
void test_typeof_expr_uninit_non_bypassed(void) {
  int n = 5;
  char a[n];
  __typeof__(id_ptr(&(int){5}), &a) bp;
  (void)bp;
}

// Verify lifetime.end runs at inner block exit (before side_effect1), not at
// outer function exit.
// CHECK-LABEL: define dso_local void @test_vla_nested_block_lifetime()
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[NB_CL:.*]])
// CHECK: call ptr @id_ptr(ptr noundef nonnull %[[NB_CL]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[NB_CL]])
// CHECK: call void @side_effect1()
void test_vla_nested_block_lifetime(void) {
  {
    char vla[*(id_ptr(&(int){5}))];
    use_vla(sizeof(vla), vla);
  }
  side_effect1();
}

// CHECK-LABEL: define dso_local void @test_vla_if_branch_lifetime(
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[IF_CL:.*]])
// CHECK: call ptr @id_ptr(ptr noundef nonnull %[[IF_CL]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[IF_CL]])
// CHECK: call void @side_effect1()
void test_vla_if_branch_lifetime(int cond) {
  if (cond)
    (void)sizeof(char[*(id_ptr(&(int){5}))]);
  side_effect1();
}

// CHECK-LABEL: define dso_local void @test_vla_param_lifetime(
// CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[PARM_CL:.*]])
// CHECK: call ptr @id_ptr(ptr noundef nonnull %[[PARM_CL]])
// CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[PARM_CL]])
void test_vla_param_lifetime(int a[*(id_ptr(&(int){5}))]) {
  (void)a;
}

int callee(int);

// Verify that conditional compound literals (which wrap CallLifetimeEnd in
// ConditionalCleanup) do not prevent [[clang::musttail]] calls.
// CHECK-LABEL: define dso_local i32 @test_musttail(
// CHECK: musttail call i32 @callee(
// CHECK-NEXT: ret i32
int test_musttail(int cond) {
  struct foo s;
  s = cond ? (struct foo){.x = 1} : (struct foo){.x = 2};
  [[clang::musttail]] return callee(s.x);
}
