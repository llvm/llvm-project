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
  MyObj* p = &x;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue (LoanID: [[L_X:[0-9]+]], ToOrigin: [[O_ADDR_X:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_ADDR_X]] (Expr: UnaryOperator))
  return p;
// CHECK:   AssignOrigin (Dest: [[O_RET_VAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_P]] (Decl: p))
// CHECK:   ReturnOfOrigin ([[O_RET_VAL]] (Expr: ImplicitCastExpr))
// CHECK:   Expire (LoanID: [[L_X]])
}


// Pointer Assignment and Return
// CHECK-LABEL: Function: assign_and_return_local_addr
// CHECK-NEXT: Block B{{[0-9]+}}:
MyObj* assign_and_return_local_addr() {
  MyObj y{20};
  MyObj* ptr1 = &y;
// CHECK: Issue (LoanID: [[L_Y:[0-9]+]], ToOrigin: [[O_ADDR_Y:[0-9]+]] (Expr: UnaryOperator))
// CHECK: AssignOrigin (Dest: [[O_PTR1:[0-9]+]] (Decl: ptr1), Src: [[O_ADDR_Y]] (Expr: UnaryOperator))
  MyObj* ptr2 = ptr1;
// CHECK: AssignOrigin (Dest: [[O_PTR1_RVAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_PTR1]] (Decl: ptr1))
// CHECK: AssignOrigin (Dest: [[O_PTR2:[0-9]+]] (Decl: ptr2), Src: [[O_PTR1_RVAL]] (Expr: ImplicitCastExpr))
  ptr2 = ptr1;
// CHECK: AssignOrigin (Dest: [[O_PTR1_RVAL_2:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_PTR1]] (Decl: ptr1))
// CHECK: AssignOrigin (Dest: [[O_PTR2]] (Decl: ptr2), Src: [[O_PTR1_RVAL_2]] (Expr: ImplicitCastExpr))
  ptr2 = ptr2; // Self assignment.
// CHECK: AssignOrigin (Dest: [[O_PTR2_RVAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_PTR2]] (Decl: ptr2))
// CHECK: AssignOrigin (Dest: [[O_PTR2]] (Decl: ptr2), Src: [[O_PTR2_RVAL]] (Expr: ImplicitCastExpr))
  return ptr2;
// CHECK: AssignOrigin (Dest: [[O_PTR2_RVAL_2:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_PTR2]] (Decl: ptr2))
// CHECK: ReturnOfOrigin ([[O_PTR2_RVAL_2]] (Expr: ImplicitCastExpr))
// CHECK: Expire (LoanID: [[L_Y]])
}

// Return of Non-Pointer Type
// CHECK-LABEL: Function: return_int_val
// CHECK-NEXT: Block B{{[0-9]+}}:
int return_int_val() {
  int x = 10;
  return x;
}
// CHECK-NEXT: End of Block


// Loan Expiration (Automatic Variable, C++)
// CHECK-LABEL: Function: loan_expires_cpp
// CHECK-NEXT: Block B{{[0-9]+}}:
void loan_expires_cpp() {
  MyObj obj{1};
  MyObj* pObj = &obj;
// CHECK: Issue (LoanID: [[L_OBJ:[0-9]+]], ToOrigin: [[O_ADDR_OBJ:[0-9]+]] (Expr: UnaryOperator))
// CHECK: AssignOrigin (Dest: [[O_POBJ:[0-9]+]] (Decl: pObj), Src: [[O_ADDR_OBJ]] (Expr: UnaryOperator))
// CHECK: Expire (LoanID: [[L_OBJ]])
}


// FIXME: No expire for Trivial Destructors
// CHECK-LABEL: Function: loan_expires_trivial
// CHECK-NEXT: Block B{{[0-9]+}}:
void loan_expires_trivial() {
  int trivial_obj = 1;
  int* pTrivialObj = &trivial_obj;
// CHECK: Issue (LoanID: [[L_TRIVIAL_OBJ:[0-9]+]], ToOrigin: [[O_ADDR_TRIVIAL_OBJ:[0-9]+]] (Expr: UnaryOperator))
// CHECK: AssignOrigin (Dest: [[O_PTOBJ:[0-9]+]] (Decl: pTrivialObj), Src: [[O_ADDR_TRIVIAL_OBJ]] (Expr: UnaryOperator))
// CHECK-NOT: Expire (LoanID: [[L_TRIVIAL_OBJ]])
// CHECK-NEXT: End of Block
  // FIXME: Add check for Expire once trivial destructors are handled for expiration.
}

// CHECK-LABEL: Function: conditional
void conditional(bool condition) {
  int a = 5;
  int b = 10;
  int* p = nullptr;

  if (condition)
    p = &a;
// CHECK: Issue (LoanID: [[L_A:[0-9]+]], ToOrigin: [[O_ADDR_A:[0-9]+]] (Expr: UnaryOperator))
// CHECK: AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_ADDR_A]] (Expr: UnaryOperator))
  else
    p = &b;
// CHECK: Issue (LoanID: [[L_B:[0-9]+]], ToOrigin: [[O_ADDR_B:[0-9]+]] (Expr: UnaryOperator))
// CHECK: AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_B]] (Expr: UnaryOperator))
  int *q = p;
// CHECK: AssignOrigin (Dest: [[O_P_RVAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_P]] (Decl: p))
// CHECK: AssignOrigin (Dest: [[O_Q:[0-9]+]] (Decl: q), Src: [[O_P_RVAL]] (Expr: ImplicitCastExpr))
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
// CHECK:   Issue (LoanID: [[L_V1:[0-9]+]], ToOrigin: [[O_ADDR_V1:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P1:[0-9]+]] (Decl: p1), Src: [[O_ADDR_V1]] (Expr: UnaryOperator))
// CHECK:   Issue (LoanID: [[L_V2:[0-9]+]], ToOrigin: [[O_ADDR_V2:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P2:[0-9]+]] (Decl: p2), Src: [[O_ADDR_V2]] (Expr: UnaryOperator))
// CHECK:   Issue (LoanID: [[L_V3:[0-9]+]], ToOrigin: [[O_ADDR_V3:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P3:[0-9]+]] (Decl: p3), Src: [[O_ADDR_V3]] (Expr: UnaryOperator))

  while (condition) {
    MyObj* temp = p1;
    p1 = p2;
    p2 = p3;
    p3 = temp;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   AssignOrigin (Dest: [[O_P1_RVAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_P1]] (Decl: p1))
// CHECK:   AssignOrigin (Dest: [[O_TEMP:[0-9]+]] (Decl: temp), Src: [[O_P1_RVAL]] (Expr: ImplicitCastExpr))
// CHECK:   AssignOrigin (Dest: [[O_P2_RVAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_P2]] (Decl: p2))
// CHECK:   AssignOrigin (Dest: [[O_P1]] (Decl: p1), Src: [[O_P2_RVAL]] (Expr: ImplicitCastExpr))
// CHECK:   AssignOrigin (Dest: [[O_P3_RVAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_P3]] (Decl: p3))
// CHECK:   AssignOrigin (Dest: [[O_P2]] (Decl: p2), Src: [[O_P3_RVAL]] (Expr: ImplicitCastExpr))
// CHECK:   AssignOrigin (Dest: [[O_TEMP_RVAL:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_TEMP]] (Decl: temp))
// CHECK:   AssignOrigin (Dest: [[O_P3]] (Decl: p3), Src: [[O_TEMP_RVAL]] (Expr: ImplicitCastExpr))
  }
}

// CHECK-LABEL: Function: overwrite_origin
void overwrite_origin() {
  MyObj s1;
  MyObj s2;
  MyObj* p = &s1;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue (LoanID: [[L_S1:[0-9]+]], ToOrigin: [[O_ADDR_S1:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_ADDR_S1]] (Expr: UnaryOperator))
  p = &s2;
// CHECK:   Issue (LoanID: [[L_S2:[0-9]+]], ToOrigin: [[O_ADDR_S2:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_S2]] (Expr: UnaryOperator))
// CHECK:   Expire (LoanID: [[L_S2]])
// CHECK:   Expire (LoanID: [[L_S1]])
}

// CHECK-LABEL: Function: reassign_to_null
void reassign_to_null() {
  MyObj s1;
  MyObj* p = &s1;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue (LoanID: [[L_S1:[0-9]+]], ToOrigin: [[O_ADDR_S1:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_ADDR_S1]] (Expr: UnaryOperator))
  p = nullptr;
// CHECK:   AssignOrigin (Dest: [[O_NULLPTR_CAST:[0-9]+]] (Expr: ImplicitCastExpr), Src: {{[0-9]+}} (Expr: CXXNullPtrLiteralExpr))
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_NULLPTR_CAST]] (Expr: ImplicitCastExpr))
// CHECK:   Expire (LoanID: [[L_S1]])
}
// FIXME: Have a better representation for nullptr than just an empty origin. 
//        It should be a separate loan and origin kind.


// CHECK-LABEL: Function: reassign_in_if
void reassign_in_if(bool condition) {
  MyObj s1;
  MyObj s2;
  MyObj* p = &s1;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue (LoanID: [[L_S1:[0-9]+]], ToOrigin: [[O_ADDR_S1:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_ADDR_S1]] (Expr: UnaryOperator))
  if (condition) {
    p = &s2;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue (LoanID: [[L_S2:[0-9]+]], ToOrigin: [[O_ADDR_S2:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_S2]] (Expr: UnaryOperator))
  }
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Expire (LoanID: [[L_S2]])
// CHECK:   Expire (LoanID: [[L_S1]])
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
      p = &s1;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue (LoanID: [[L_S1:[0-9]+]], ToOrigin: [[O_ADDR_S1:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_S1]] (Expr: UnaryOperator))
      break;
    case 2:
      p = &s2;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue (LoanID: [[L_S2:[0-9]+]], ToOrigin: [[O_ADDR_S2:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_S2]] (Expr: UnaryOperator))
      break;
    default:
      p = &s3;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue (LoanID: [[L_S3:[0-9]+]], ToOrigin: [[O_ADDR_S3:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_S3]] (Expr: UnaryOperator))
      break;
  }
// CHECK: Block B{{[0-9]+}}:
// CHECK-DAG:   Expire (LoanID: [[L_S3]])
// CHECK-DAG:   Expire (LoanID: [[L_S2]])
// CHECK-DAG:   Expire (LoanID: [[L_S1]])
}

// CHECK-LABEL: Function: loan_in_loop
void loan_in_loop(bool condition) {
  MyObj* p = nullptr;
  // CHECK:   AssignOrigin (Dest: [[O_NULLPTR_CAST:[0-9]+]] (Expr: ImplicitCastExpr), Src: [[O_NULLPTR:[0-9]+]] (Expr: CXXNullPtrLiteralExpr))
  // CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_NULLPTR_CAST]] (Expr: ImplicitCastExpr))
  while (condition) {
    MyObj inner;
    p = &inner;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue (LoanID: [[L_INNER:[0-9]+]], ToOrigin: [[O_ADDR_INNER:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_INNER]] (Expr: UnaryOperator))
// CHECK:   Expire (LoanID: [[L_INNER]])
  }
}

// CHECK-LABEL: Function: loop_with_break
void loop_with_break(int count) {
  MyObj s1;
  MyObj s2;
  MyObj* p = &s1;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue (LoanID: [[L_S1:[0-9]+]], ToOrigin: [[O_ADDR_S1:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_ADDR_S1]] (Expr: UnaryOperator))
  for (int i = 0; i < count; ++i) {
    if (i == 5) {
      p = &s2;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue (LoanID: [[L_S2:[0-9]+]], ToOrigin: [[O_ADDR_S2:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_S2]] (Expr: UnaryOperator))
      break;
    }
  }
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Expire (LoanID: [[L_S2]])
// CHECK:   Expire (LoanID: [[L_S1]])
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
// CHECK:   Issue (LoanID: [[L_OUTER:[0-9]+]], ToOrigin: [[O_ADDR_OUTER:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_OUTER]] (Expr: UnaryOperator))
    {
      MyObj inner;
      p = &inner;
// CHECK:   Issue (LoanID: [[L_INNER:[0-9]+]], ToOrigin: [[O_ADDR_INNER:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P]] (Decl: p), Src: [[O_ADDR_INNER]] (Expr: UnaryOperator))
    }
// CHECK:   Expire (LoanID: [[L_INNER]])
  }
// CHECK:   Expire (LoanID: [[L_OUTER]])
}

// CHECK-LABEL: Function: pointer_indirection
void pointer_indirection() {
  int a;
  int *p = &a;
// CHECK: Block B1:
// CHECK:   Issue (LoanID: [[L_A:[0-9]+]], ToOrigin: [[O_ADDR_A:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_P:[0-9]+]] (Decl: p), Src: [[O_ADDR_A]] (Expr: UnaryOperator))
  int **pp = &p;
// CHECK:   Issue (LoanID: [[L_P:[0-g]+]], ToOrigin: [[O_ADDR_P:[0-9]+]] (Expr: UnaryOperator))
// CHECK:   AssignOrigin (Dest: [[O_PP:[0-9]+]] (Decl: pp), Src: [[O_ADDR_P]] (Expr: UnaryOperator))

// FIXME: The Origin for the RHS is broken
  int *q = *pp;
// CHECK:   AssignOrigin (Dest: {{[0-9]+}} (Decl: q), Src: {{[0-9]+}} (Expr: ImplicitCastExpr))
}
