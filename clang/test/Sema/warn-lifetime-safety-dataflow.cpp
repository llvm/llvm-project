// RUN: %clang_cc1 -fexperimental-lifetime-safety -mllvm -debug-only=LifetimeFacts -Wexperimental-lifetime-safety %s 2>&1 | FileCheck %s
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
// CHECK:   Issue ([[L_X:[0-9]+]] (Path: x), ToOrigin: [[O_DRE_X:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_X:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_X]] (Expr: DeclRefExpr))
  MyObj* p = &x;
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_ADDR_X]] (Expr: UnaryOperator))
  return p;
// CHECK:   Use ([[O_P]] (Decl: p), Read)
// CHECK:   AssignOrigin (Dest: [[O_RET_VAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_P]] (Decl: p))
// CHECK:   ReturnOfOrigin ([[O_RET_VAL]] (Expr: ImplicitCastExpr))
// CHECK:   Expire ([[L_X]] (Path: x))
}


// Pointer Assignment and Return
// CHECK-LABEL: Function: assign_and_return_local_addr
MyObj* assign_and_return_local_addr() {
  MyObj y{20};
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_Y:[0-9]+]] (Path: y), ToOrigin: [[O_DRE_Y:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_Y:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_Y]] (Expr: DeclRefExpr))
  MyObj* ptr1 = &y;
// CHECK:   AssignOrigin (Dest: [[O_PTR1:[0-9]+]] (Decl: ptr1), Src: [[O_ADDR_Y]] (Expr: UnaryOperator))
  MyObj* ptr2 = ptr1;
// CHECK:   Use ([[O_PTR1]] (Decl: ptr1), Read)
// CHECK:   AssignOrigin (Dest: [[O_PTR1_RVAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_PTR1]] (Decl: ptr1))
// CHECK:   AssignOrigin (Dest: [[O_PTR2:[0-9]+]] (Decl: ptr2), Src: [[O_PTR1_RVAL]] (Expr: ImplicitCastExpr))
  ptr2 = ptr1;
// CHECK:   Use ([[O_PTR1]] (Decl: ptr1), Read)
// CHECK:   AssignOrigin (Dest: [[O_PTR1_RVAL_2:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_PTR1]] (Decl: ptr1))
// CHECK:   Use ({{[0-9]+}} (Decl: ptr2), Write)
// CHECK:   AssignOrigin (Dest: [[O_PTR2]] (Decl: ptr2), Src: [[O_PTR1_RVAL_2]] (Expr: ImplicitCastExpr))
  ptr2 = ptr2; // Self assignment.
// CHECK:   Use ([[O_PTR2]] (Decl: ptr2), Read)
// CHECK:   AssignOrigin (Dest: [[O_PTR2_RVAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_PTR2]] (Decl: ptr2))
// CHECK:   Use ([[O_PTR2]] (Decl: ptr2), Write)
// CHECK:   AssignOrigin (Dest: [[O_PTR2]] (Decl: ptr2), Src: [[O_PTR2_RVAL]] (Expr: ImplicitCastExpr))
  return ptr2;
// CHECK:   Use ([[O_PTR2]] (Decl: ptr2), Read)
// CHECK:   AssignOrigin (Dest: [[O_RET_VAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_PTR2]] (Decl: ptr2))
// CHECK:   ReturnOfOrigin ([[O_RET_VAL]] (Expr: ImplicitCastExpr))
// CHECK:   Expire ([[L_Y]] (Path: y))
}

// Return of Non-Pointer Type
// CHECK-LABEL: Function: return_int_val
int return_int_val() {
  int x = 10;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_X:[0-9]+]] (Path: x), ToOrigin: {{[0-9]+}} (Expr: DeclRefExpr))
  return x;
}
// CHECK-NEXT: End of Block


// Loan Expiration (Automatic Variable, C++)
// CHECK-LABEL: Function: loan_expires_cpp
void loan_expires_cpp() {
  MyObj obj{1};
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_OBJ:[0-9]+]] (Path: obj), ToOrigin: [[O_DRE_OBJ:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_OBJ:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_OBJ]] (Expr: DeclRefExpr))
  MyObj* pObj = &obj;
// CHECK:   AssignOrigin (Dest: {{[0-9]+}} (Decl: pObj), Src: [[O_ADDR_OBJ]] (Expr: UnaryOperator))
// CHECK:   Expire ([[L_OBJ]] (Path: obj))
}


// FIXME: No expire for Trivial Destructors
// CHECK-LABEL: Function: loan_expires_trivial
void loan_expires_trivial() {
  int trivial_obj = 1;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_TRIVIAL_OBJ:[0-9]+]] (Path: trivial_obj), ToOrigin: [[O_DRE_TRIVIAL:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_TRIVIAL_OBJ:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_TRIVIAL]] (Expr: DeclRefExpr))
  int* pTrivialObj = &trivial_obj;
// CHECK:   AssignOrigin (Dest: {{[0-9]+}} (Decl: pTrivialObj), Src: [[O_ADDR_TRIVIAL_OBJ]] (Expr: UnaryOperator))
// CHECK-NOT: Expire
// CHECK-NEXT: End of Block
  // FIXME: Add check for Expire once trivial destructors are handled for expiration.
}

// CHECK-LABEL: Function: conditional
void conditional(bool condition) {
  int a = 5;
  int b = 10;
  int* p = nullptr;

  if (condition)
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_A:[0-9]+]] (Path: a), ToOrigin: [[O_DRE_A:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_A:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_A]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_ADDR_A]] (Expr: UnaryOperator))
    p = &a;
  else
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_B:[0-9]+]] (Path: b), ToOrigin: [[O_DRE_B:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_B:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_B]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_ADDR_B]] (Expr: UnaryOperator))
    p = &b;
// CHECK: Block B{{[0-9]+}}:
  int *q = p;
// CHECK:   Use ([[O_P]] (Decl: p), Read)
// CHECK:   AssignOrigin (Dest: [[O_P_RVAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_P]] (Decl: p))
// CHECK:   AssignOrigin (Dest: [[O_Q:[0-9]+]] (Decl: q), Src: [[O_P_RVAL]] (Expr: ImplicitCastExpr))
}


// CHECK-LABEL: Function: pointers_in_a_cycle
void pointers_in_a_cycle(bool condition) {
  MyObj v1{1};
  MyObj v2{1};
  MyObj v3{1};

  MyObj* p1 = &v1;
  MyObj* p2 = &v2;
  MyObj* p3 = &v3;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_V1:[0-9]+]] (Path: v1), ToOrigin: [[O_DRE_V1:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_V1:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_V1]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_P1:[0-9]+]] (Decl: p1), Src: [[O_ADDR_V1]] (Expr: UnaryOperator))
// CHECK:   Issue ([[L_V2:[0-9]+]] (Path: v2), ToOrigin: [[O_DRE_V2:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_V2:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_V2]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_P2:[0-9]+]] (Decl: p2), Src: [[O_ADDR_V2]] (Expr: UnaryOperator))
// CHECK:   Issue ([[L_V3:[0-9]+]] (Path: v3), ToOrigin: [[O_DRE_V3:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_V3:[0-g]+]] (Expr: UnaryOperator), Src: [[O_DRE_V3]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_P3:[0-9]+]] (Decl: p3), Src: [[O_ADDR_V3]] (Expr: UnaryOperator))

  while (condition) {
// CHECK: Block B{{[0-9]+}}:
    MyObj* temp = p1;
// CHECK:   Use ([[O_P1]] (Decl: p1), Read)
// CHECK:   AssignOrigin (Dest: [[O_P1_RVAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_P1]] (Decl: p1))
// CHECK:   AssignOrigin (Dest: [[O_TEMP:[0-9]+]] (Decl: temp), Src: [[O_P1_RVAL]] (Expr: ImplicitCastExpr))
    p1 = p2;
// CHECK:   Use ([[O_P2:[0-9]+]] (Decl: p2), Read)
// CHECK:   AssignOrigin (Dest: [[O_P2_RVAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_P2]] (Decl: p2))
// CHECK:   Use ({{[0-9]+}} (Decl: p1), Write)
// CHECK:   AssignOrigin (Dest: [[O_P1]] (Decl: p1), Src: [[O_P2_RVAL]] (Expr: ImplicitCastExpr))
    p2 = p3;
// CHECK:   Use ([[O_P3:[0-9]+]] (Decl: p3), Read)
// CHECK:   AssignOrigin (Dest: [[O_P3_RVAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_P3]] (Decl: p3))
// CHECK:   Use ({{[0-9]+}} (Decl: p2), Write)
// CHECK:   AssignOrigin (Dest: [[O_P2]] (Decl: p2), Src: [[O_P3_RVAL]] (Expr: ImplicitCastExpr))
    p3 = temp;
// CHECK:   Use ([[O_TEMP]] (Decl: temp), Read)
// CHECK:   AssignOrigin (Dest: [[O_TEMP_RVAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_TEMP]] (Decl: temp))
// CHECK:   Use ({{[0-9]+}} (Decl: p3), Write)
// CHECK:   AssignOrigin (Dest: [[O_P3]] (Decl: p3), Src: [[O_TEMP_RVAL]] (Expr: ImplicitCastExpr))
  }
}

// CHECK-LABEL: Function: overwrite_origin
void overwrite_origin() {
  MyObj s1;
  MyObj s2;
// CHECK: Block B{{[0-9]+}}:
  MyObj* p = &s1;
// CHECK:   Issue ([[L_S1:[0-9]+]] (Path: s1), ToOrigin: [[O_DRE_S1:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_S1:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_S1]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_ADDR_S1]] (Expr: UnaryOperator))
  p = &s2;
// CHECK:   Issue ([[L_S2:[0-9]+]] (Path: s2), ToOrigin: [[O_DRE_S2:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_S2:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_S2]] (Expr: DeclRefExpr))
// CHECK:   Use ({{[0-9]+}} (Decl: p), Write)
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_S2]] (Expr: UnaryOperator))
// CHECK:   Expire ([[L_S2]] (Path: s2))
// CHECK:   Expire ([[L_S1]] (Path: s1))
}

// CHECK-LABEL: Function: reassign_to_null
void reassign_to_null() {
  MyObj s1;
// CHECK: Block B{{[0-9]+}}:
  MyObj* p = &s1;
// CHECK:   Issue ([[L_S1:[0-9]+]] (Path: s1), ToOrigin: [[O_DRE_S1:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_S1:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_S1]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_ADDR_S1]] (Expr: UnaryOperator))
  p = nullptr;
// CHECK:   AssignOrigin (Dest: [[O_NULLPTR_CAST:[0-9]+]] (Expr: ImplicitCastExpr), Src: {{[0-9]+}} (Expr: CXXNullPtrLiteralExpr))
// CHECK:   Use ({{[0-9]+}} (Decl: p), Write)
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_NULLPTR_CAST]] (Expr: ImplicitCastExpr))
// CHECK:   Expire ([[L_S1]] (Path: s1))
}
// FIXME: Have a better representation for nullptr than just an empty origin. 
//        It should be a separate loan and origin kind.


// CHECK-LABEL: Function: reassign_in_if
void reassign_in_if(bool condition) {
  MyObj s1;
  MyObj s2;
  MyObj* p = &s1;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_S1:[0-9]+]] (Path: s1), ToOrigin: [[O_DRE_S1:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_S1:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_S1]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_ADDR_S1]] (Expr: UnaryOperator))
  if (condition) {
// CHECK: Block B{{[0-9]+}}:
    p = &s2;
// CHECK:   Issue ([[L_S2:[0-9]+]] (Path: s2), ToOrigin: [[O_DRE_S2:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_S2:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_S2]] (Expr: DeclRefExpr))
// CHECK:   Use ({{[0-9]+}} (Decl: p), Write)
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_S2]] (Expr: UnaryOperator))
  }
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Expire ([[L_S2]] (Path: s2))
// CHECK:   Expire ([[L_S1]] (Path: s1))
}


// CHECK-LABEL: Function: assign_in_switch
void assign_in_switch(int mode) {
  MyObj s1;
  MyObj s2;
  MyObj s3;
  MyObj* p = nullptr;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   AssignOrigin (Dest: [[O_NULLPTR_CAST:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_NULLPTR:[0-9]+]] (Expr: CXXNullPtrLiteralExpr))
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_NULLPTR_CAST]] (Expr: ImplicitCastExpr))
  switch (mode) {
    case 1:
// CHECK-DAG: Block B{{[0-9]+}}:
      p = &s1;
// CHECK-DAG:   Issue ([[L_S1:[0-9]+]] (Path: s1), ToOrigin: [[O_DRE_S1:[0-9]+]] (Expr: DeclRefExpr))
// CHECK-DAG:   AssignOrigin (Dest: [[O_ADDR_S1:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_S1]] (Expr: DeclRefExpr))
// CHECK-DAG:   Use ({{[0-9]+}} (Decl: p), Write)
// CHECK-DAG:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_S1]] (Expr: UnaryOperator))
      break;
    case 2:
// CHECK-DAG: Block B{{[0-9]+}}:
      p = &s2;
// CHECK-DAG:   Issue ([[L_S2:[0-9]+]] (Path: s2), ToOrigin: [[O_DRE_S2:[0-9]+]] (Expr: DeclRefExpr))
// CHECK-DAG:   AssignOrigin (Dest: [[O_ADDR_S2:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_S2]] (Expr: DeclRefExpr))
// CHECK-DAG:   Use ({{[0-9]+}} (Decl: p), Write)
// CHECK-DAG:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_S2]] (Expr: UnaryOperator))
      break;
    default:
// CHECK: Block B{{[0-9]+}}:
      p = &s3;
// CHECK:   Issue ([[L_S3:[0-9]+]] (Path: s3), ToOrigin: [[O_DRE_S3:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_S3:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_S3]] (Expr: DeclRefExpr))
// CHECK:   Use ({{[0-9]+}} (Decl: p), Write)
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_S3]] (Expr: UnaryOperator))
      break;
  }
// CHECK: Block B{{[0-9]+}}:
// CHECK-DAG:   Expire ([[L_S3]] (Path: s3))
// CHECK-DAG:   Expire ([[L_S2]] (Path: s2))
// CHECK-DAG:   Expire ([[L_S1]] (Path: s1))
}

// CHECK-LABEL: Function: loan_in_loop
void loan_in_loop(bool condition) {
  MyObj* p = nullptr;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   AssignOrigin (Dest: [[O_NULLPTR_CAST:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_NULLPTR:[0-9]+]] (Expr: CXXNullPtrLiteralExpr))
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_NULLPTR_CAST]] (Expr: ImplicitCastExpr))
  while (condition) {
    MyObj inner;
// CHECK: Block B{{[0-9]+}}:
    p = &inner;
// CHECK:   Issue ([[L_INNER:[0-9]+]] (Path: inner), ToOrigin: [[O_DRE_INNER:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_INNER:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_INNER]] (Expr: DeclRefExpr))
// CHECK:   Use ({{[0-9]+}} (Decl: p), Write)
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_INNER]] (Expr: UnaryOperator))
// CHECK:   Expire ([[L_INNER]] (Path: inner))
  }
}

// CHECK-LABEL: Function: loop_with_break
void loop_with_break(int count) {
  MyObj s1;
  MyObj s2;
  MyObj* p = &s1;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_S1:[0-9]+]] (Path: s1), ToOrigin: [[O_DRE_S1:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_S1:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_S1]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_ADDR_S1]] (Expr: UnaryOperator))
  for (int i = 0; i < count; ++i) {
    if (i == 5) {
// CHECK: Block B{{[0-9]+}}:
      p = &s2;
// CHECK:   Issue ([[L_S2:[0-9]+]] (Path: s2), ToOrigin: [[O_DRE_S2:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_S2:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_S2]] (Expr: DeclRefExpr))
// CHECK:   Use ({{[0-9]+}} (Decl: p), Write)
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_S2]] (Expr: UnaryOperator))
      break;
    }
  }
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Expire ([[L_S2]] (Path: s2))
// CHECK:   Expire ([[L_S1]] (Path: s1))
}

// CHECK-LABEL: Function: nested_scopes
void nested_scopes() {
  MyObj* p = nullptr;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   AssignOrigin (Dest: [[O_NULLPTR_CAST:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_NULLPTR:[0-9]+]] (Expr: CXXNullPtrLiteralExpr))
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_NULLPTR_CAST]] (Expr: ImplicitCastExpr))
  {
    MyObj outer;
    p = &outer;
// CHECK:   Issue ([[L_OUTER:[0-9]+]] (Path: outer), ToOrigin: [[O_DRE_OUTER:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_OUTER:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_OUTER]] (Expr: DeclRefExpr))
// CHECK:   Use ({{[0-9]+}} (Decl: p), Write)
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_OUTER]] (Expr: UnaryOperator))
    {
      MyObj inner;
      p = &inner;
// CHECK:   Issue ([[L_INNER:[0-9]+]] (Path: inner), ToOrigin: [[O_DRE_INNER:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_INNER:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_INNER]] (Expr: DeclRefExpr))
// CHECK:   Use ({{[0-9]+}} (Decl: p), Write)
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_INNER]] (Expr: UnaryOperator))
    }
// CHECK:   Expire ([[L_INNER]] (Path: inner))
  }
// CHECK:   Expire ([[L_OUTER]] (Path: outer))
}

// CHECK-LABEL: Function: pointer_indirection
void pointer_indirection() {
  int a;
  int *p = &a;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_A:[0-9]+]] (Path: a), ToOrigin: [[O_DRE_A:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_A:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_A]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_ADDR_A]] (Expr: UnaryOperator))
  int **pp = &p;
// Note: No facts are generated for &p because the subexpression is a pointer type,
// which is not yet supported by the origin model. This is expected.
  int *q = *pp;
// CHECK:   Use ([[O_PP:[0-9]+]] (Decl: pp), Read)
// CHECK:   AssignOrigin (Dest: {{[0-9]+}} (Decl: q), Src: {{[0-9]+}} (Expr: ImplicitCastExpr))
}

// CHECK-LABEL: Function: ternary_operator
// FIXME: Propagate origins across ConditionalOperator.
void ternary_operator() {
  int a, b;
  int *p;
  p = (a > b) ? &a : &b;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_A:[0-9]+]] (Path: a), ToOrigin: [[O_DRE_A:[0-9]+]] (Expr: DeclRefExpr))

// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue ([[L_B:[0-9]+]] (Path: b), ToOrigin: [[O_DRE_B:[0-9]+]] (Expr: DeclRefExpr))

// CHECK: Block B{{[0-9]+}}:
// CHECK:   Use ({{[0-9]+}} (Decl: p), Write)
// CHECK:   AssignOrigin (Dest: {{[0-9]+}} (Decl: p), Src: {{[0-9]+}} (Expr: ConditionalOperator))
}

// CHECK-LABEL: Function: test_use_facts
void usePointer(MyObj*);
void test_use_facts() {
  MyObj x;
  MyObj *p;
// CHECK: Block B{{[0-9]+}}:
  p = &x;
// CHECK:   Issue ([[L_X:[0-9]+]] (Path: x), ToOrigin: [[O_DRE_X:[0-9]+]] (Expr: DeclRefExpr))
// CHECK:   AssignOrigin (Dest: [[O_ADDR_X:[0-9]+]] (Expr: UnaryOperator), Src: [[O_DRE_X]] (Expr: DeclRefExpr))
// CHECK:   Use ([[O_P:[0-9]+]] (Decl: p), Write)
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_X]] (Expr: UnaryOperator))
  (void)*p;
// CHECK:   Use ([[O_P]] (Decl: p), Read)
  usePointer(p);
// CHECK:   Use ([[O_P]] (Decl: p), Read)
  p->id = 1;
// CHECK:   Use ([[O_P]] (Decl: p), Read)
  MyObj* q;
  q = p;
// CHECK:   Use ([[O_P]] (Decl: p), Read)
// CHECK:   Use ([[O_Q:[0-9]+]] (Decl: q), Write)
  usePointer(q);
// CHECK:   Use ([[O_Q]] (Decl: q), Read)
  q->id = 2;
// CHECK:   Use ([[O_Q]] (Decl: q), Read)
// CHECK:   Expire ([[L_X]] (Path: x))
}