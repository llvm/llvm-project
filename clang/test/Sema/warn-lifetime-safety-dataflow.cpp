// RUN: %clang_cc1 -mllvm -debug-only=LifetimeFacts -Wlifetime-safety %s 2>&1 | FileCheck %s
// REQUIRES: asserts

struct MyObj {
  int id;
  ~MyObj() {} // Non-trivial destructor
};

// Simple Local Variable Address and Return
// CHECK-LABEL: Function: return_local_addr
MyObj* return_local_addr() {
  MyObj x {10};
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_X:[0-9]+]] (Path: x), ToOrigin: [[O_DRE_X:[0-9]+]] (Expr: DeclRefExpr, Decl: x))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_ADDR_X:[0-9]+]] (Expr: UnaryOperator, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_DRE_X]] (Expr: DeclRefExpr, Decl: x)
  MyObj* p = &x;
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_P:[0-9]+]] (Decl: p, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_ADDR_X]] (Expr: UnaryOperator, Type : MyObj *)
// CHECK:   Use ([[O_P]] (Decl: p, Type : MyObj *), Read)
  return p;
// CHECK:   Issue ({{[0-9]+}} (Path: p), ToOrigin: {{[0-9]+}} (Expr: DeclRefExpr, Decl: p))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_RET_VAL:[0-9]+]] (Expr: ImplicitCastExpr, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_P]] (Decl: p, Type : MyObj *)
// CHECK:   Expire ([[L_X]] (Path: x))
// CHECK:   Expire ({{[0-9]+}} (Path: p))
// CHECK:   OriginEscapes ([[O_RET_VAL]] (Expr: ImplicitCastExpr, Type : MyObj *))
}

// Loan Expiration (Automatic Variable, C++)
// CHECK-LABEL: Function: loan_expires_cpp
void loan_expires_cpp() {
  MyObj obj{1};
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_OBJ:[0-9]+]] (Path: obj), ToOrigin: [[O_DRE_OBJ:[0-9]+]] (Expr: DeclRefExpr, Decl: obj))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_ADDR_OBJ:[0-9]+]] (Expr: UnaryOperator, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_DRE_OBJ]] (Expr: DeclRefExpr, Decl: obj)
  MyObj* pObj = &obj;
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: {{[0-9]+}} (Decl: pObj, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_ADDR_OBJ]] (Expr: UnaryOperator, Type : MyObj *)
// CHECK:   Expire ([[L_OBJ]] (Path: obj))
}


// CHECK-LABEL: Function: loan_expires_trivial
void loan_expires_trivial() {
  int trivial_obj = 1;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_TRIVIAL_OBJ:[0-9]+]] (Path: trivial_obj), ToOrigin: [[O_DRE_TRIVIAL:[0-9]+]] (Expr: DeclRefExpr, Decl: trivial_obj))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_ADDR_TRIVIAL_OBJ:[0-9]+]] (Expr: UnaryOperator, Type : int *)
// CHECK-NEXT:       Src:  [[O_DRE_TRIVIAL]] (Expr: DeclRefExpr, Decl: trivial_obj)
  int* pTrivialObj = &trivial_obj;
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: {{[0-9]+}} (Decl: pTrivialObj, Type : int *)
// CHECK-NEXT:       Src:  [[O_ADDR_TRIVIAL_OBJ]] (Expr: UnaryOperator, Type : int *)
// CHECK:   Expire ([[L_TRIVIAL_OBJ]] (Path: trivial_obj))
// CHECK-NEXT: End of Block
}

// CHECK-LABEL: Function: overwrite_origin
void overwrite_origin() {
  MyObj s1;
  MyObj s2;
// CHECK: Block B{{[0-9]+}}:
  MyObj* p = &s1;
// CHECK:   Issue ([[L_S1:[0-9]+]] (Path: s1), ToOrigin: [[O_DRE_S1:[0-9]+]] (Expr: DeclRefExpr, Decl: s1))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_ADDR_S1:[0-9]+]] (Expr: UnaryOperator, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_DRE_S1]] (Expr: DeclRefExpr, Decl: s1)
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_P:[0-9]+]] (Decl: p, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_ADDR_S1]] (Expr: UnaryOperator, Type : MyObj *)
  p = &s2;
// CHECK:   Issue ([[L_S2:[0-9]+]] (Path: s2), ToOrigin: [[O_DRE_S2:[0-9]+]] (Expr: DeclRefExpr, Decl: s2))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_ADDR_S2:[0-9]+]] (Expr: UnaryOperator, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_DRE_S2]] (Expr: DeclRefExpr, Decl: s2)
// CHECK:   Use ([[O_P]] (Decl: p, Type : MyObj *), Write)
// CHECK:   Issue ({{[0-9]+}} (Path: p), ToOrigin: {{[0-9]+}} (Expr: DeclRefExpr, Decl: p))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_P]] (Decl: p, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_ADDR_S2]] (Expr: UnaryOperator, Type : MyObj *)
// CHECK:   Expire ([[L_S2]] (Path: s2))
// CHECK:   Expire ([[L_S1]] (Path: s1))
}

// CHECK-LABEL: Function: reassign_to_null
void reassign_to_null() {
  MyObj s1;
// CHECK: Block B{{[0-9]+}}:
  MyObj* p = &s1;
// CHECK:   Issue ([[L_S1:[0-9]+]] (Path: s1), ToOrigin: [[O_DRE_S1:[0-9]+]] (Expr: DeclRefExpr, Decl: s1))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_ADDR_S1:[0-9]+]] (Expr: UnaryOperator, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_DRE_S1]] (Expr: DeclRefExpr, Decl: s1)
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_P:[0-9]+]] (Decl: p, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_ADDR_S1]] (Expr: UnaryOperator, Type : MyObj *)
  p = nullptr;
// CHECK:   Use ([[O_P]] (Decl: p, Type : MyObj *), Write)
// CHECK:   Issue ({{[0-9]+}} (Path: p), ToOrigin: {{[0-9]+}} (Expr: DeclRefExpr, Decl: p))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_P]] (Decl: p, Type : MyObj *)
// CHECK-NEXT:       Src:  {{[0-9]+}} (Expr: ImplicitCastExpr, Type : MyObj *)
// CHECK:   Expire ([[L_S1]] (Path: s1))
}
// FIXME: Have a better representation for nullptr than just an empty origin. 
//        It should be a separate loan and origin kind.

// CHECK-LABEL: Function: pointer_indirection
void pointer_indirection() {
  int a;
  int *p = &a;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_A:[0-9]+]] (Path: a), ToOrigin: [[O_DRE_A:[0-9]+]] (Expr: DeclRefExpr, Decl: a))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_ADDR_A:[0-9]+]] (Expr: UnaryOperator, Type : int *)
// CHECK-NEXT:       Src:  [[O_DRE_A]] (Expr: DeclRefExpr, Decl: a)
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_P:[0-9]+]] (Decl: p, Type : int *)
// CHECK-NEXT:       Src:  [[O_ADDR_A]] (Expr: UnaryOperator, Type : int *)
  int **pp = &p;
// CHECK:   Use ([[O_P]] (Decl: p, Type : int *), Read)
// CHECK:   Issue ({{[0-9]+}} (Path: p), ToOrigin: {{[0-9]+}} (Expr: DeclRefExpr, Decl: p))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: {{[0-9]+}} (Expr: UnaryOperator, Type : int **)
// CHECK-NEXT:       Src:  {{[0-9]+}} (Expr: DeclRefExpr, Decl: p)
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: {{[0-9]+}} (Expr: UnaryOperator, Type : int *)
// CHECK-NEXT:       Src:  [[O_P]] (Decl: p, Type : int *)
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_PP_OUTER:[0-9]+]] (Decl: pp, Type : int **)
// CHECK-NEXT:       Src:  {{[0-9]+}} (Expr: UnaryOperator, Type : int **)
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_PP_INNER:[0-9]+]] (Decl: pp, Type : int *)
// CHECK-NEXT:       Src:  {{[0-9]+}} (Expr: UnaryOperator, Type : int *)
  
  // FIXME: Propagate origins across dereference unary operator*
  int *q = *pp;
// CHECK:   Use ([[O_PP_OUTER]] (Decl: pp, Type : int **), [[O_PP_INNER]] (Decl: pp, Type : int *), Read)
// CHECK:   Issue ({{[0-9]+}} (Path: pp), ToOrigin: {{[0-9]+}} (Expr: DeclRefExpr, Decl: pp))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: {{[0-9]+}} (Expr: ImplicitCastExpr, Type : int **)
// CHECK-NEXT:       Src:  [[O_PP_OUTER]] (Decl: pp, Type : int **)
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: {{[0-9]+}} (Expr: ImplicitCastExpr, Type : int *)
// CHECK-NEXT:       Src:  [[O_PP_INNER]] (Decl: pp, Type : int *)
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: {{[0-9]+}} (Expr: UnaryOperator, Type : int *&)
// CHECK-NEXT:       Src:  {{[0-9]+}} (Expr: ImplicitCastExpr, Type : int **)
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: {{[0-9]+}} (Expr: UnaryOperator, Type : int *)
// CHECK-NEXT:       Src:  {{[0-9]+}} (Expr: ImplicitCastExpr, Type : int *)
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: {{[0-9]+}} (Expr: ImplicitCastExpr, Type : int *)
// CHECK-NEXT:       Src:  {{[0-9]+}} (Expr: UnaryOperator, Type : int *)
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: {{[0-9]+}} (Decl: q, Type : int *)
// CHECK-NEXT:       Src:  {{[0-9]+}} (Expr: ImplicitCastExpr, Type : int *)
}

// CHECK-LABEL: Function: test_use_lifetimebound_call
MyObj* LifetimeBoundCall(MyObj* x [[clang::lifetimebound]], MyObj* y [[clang::lifetimebound]]);
void test_use_lifetimebound_call() {
  MyObj x, y;
  MyObj *p = &x;
// CHECK:   Issue ([[L_X:[0-9]+]] (Path: x), ToOrigin: [[O_DRE_X:[0-9]+]] (Expr: DeclRefExpr, Decl: x))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_ADDR_X:[0-9]+]] (Expr: UnaryOperator, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_DRE_X]] (Expr: DeclRefExpr, Decl: x)
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_P:[0-9]+]] (Decl: p, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_ADDR_X]] (Expr: UnaryOperator, Type : MyObj *)
  MyObj *q = &y;
// CHECK:   Issue ([[L_Y:[0-9]+]] (Path: y), ToOrigin: [[O_DRE_Y:[0-9]+]] (Expr: DeclRefExpr, Decl: y))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_ADDR_Y:[0-9]+]] (Expr: UnaryOperator, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_DRE_Y]] (Expr: DeclRefExpr, Decl: y)
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_Q:[0-9]+]] (Decl: q, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_ADDR_Y]] (Expr: UnaryOperator, Type : MyObj *)
  MyObj* r = LifetimeBoundCall(p, q);
// CHECK:   Use ([[O_P]] (Decl: p, Type : MyObj *), Read)
// CHECK:   Issue ({{[0-9]+}} (Path: p), ToOrigin: {{[0-9]+}} (Expr: DeclRefExpr, Decl: p))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_P_RVAL:[0-9]+]] (Expr: ImplicitCastExpr, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_P]] (Decl: p, Type : MyObj *)
// CHECK:   Use ([[O_Q]] (Decl: q, Type : MyObj *), Read)
// CHECK:   Issue ({{[0-9]+}} (Path: q), ToOrigin: {{[0-9]+}} (Expr: DeclRefExpr, Decl: q))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_Q_RVAL:[0-9]+]] (Expr: ImplicitCastExpr, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_Q]] (Decl: q, Type : MyObj *)
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_CALL_EXPR:[0-9]+]] (Expr: CallExpr, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_P_RVAL]] (Expr: ImplicitCastExpr, Type : MyObj *)
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_CALL_EXPR]] (Expr: CallExpr, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_Q_RVAL]] (Expr: ImplicitCastExpr, Type : MyObj *), Merge
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: {{[0-9]+}} (Decl: r, Type : MyObj *)
// CHECK-NEXT:       Src:  [[O_CALL_EXPR]] (Expr: CallExpr, Type : MyObj *)
// CHECK:   Expire ([[L_Y]] (Path: y))
// CHECK:   Expire ([[L_X]] (Path: x))
}

// CHECK-LABEL: Function: test_reference_variable
void test_reference_variable() {
  MyObj x;
  const MyObj* p;
  const MyObj& y = x;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_X:[0-9]+]] (Path: x), ToOrigin: [[O_DRE_X:[0-9]+]] (Expr: DeclRefExpr, Decl: x))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_CAST_Y:[0-9]+]] (Expr: ImplicitCastExpr, Type : const MyObj &)
// CHECK-NEXT:       Src:  [[O_DRE_X]] (Expr: DeclRefExpr, Decl: x)
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_Y:[0-9]+]] (Decl: y, Type : const MyObj &)
// CHECK-NEXT:       Src:  [[O_CAST_Y]] (Expr: ImplicitCastExpr, Type : const MyObj &)
  const MyObj& z = y;
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: [[O_Z:[0-9]+]] (Decl: z, Type : const MyObj &)
// CHECK-NEXT:       Src:  [[O_Y]] (Decl: y, Type : const MyObj &)
  p = &z;
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: {{[0-9]+}} (Expr: UnaryOperator, Type : const MyObj *)
// CHECK-NEXT:       Src:  [[O_Z]] (Decl: z, Type : const MyObj &)
// CHECK:   Use ({{[0-9]+}} (Decl: p, Type : const MyObj *), Write)
// CHECK:   Issue ({{[0-9]+}} (Path: p), ToOrigin: {{[0-9]+}} (Expr: DeclRefExpr, Decl: p))
// CHECK:   OriginFlow:
// CHECK-NEXT:       Dest: {{[0-9]+}} (Decl: p, Type : const MyObj *)
// CHECK-NEXT:       Src:  {{[0-9]+}} (Expr: UnaryOperator, Type : const MyObj *)
// CHECK:   Expire ([[L_X]] (Path: x))
}
