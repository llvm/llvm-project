// RUN: %clang_cc1 -mllvm -debug-only=LifetimeFacts -Wexperimental-lifetime-safety %s 2>&1 | FileCheck %s
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
// CHECK:   Issue (LoanID: [[L_X:[0-9]+]], OriginID: [[O_ADDR_X:[0-9]+]])
// CHECK:   AssignOrigin (DestID: [[O_P:[0-9]+]], SrcID: [[O_ADDR_X]])
  return p;
// CHECK:   AssignOrigin (DestID: [[O_RET_VAL:[0-9]+]], SrcID: [[O_P]])
// CHECK:   ReturnOfOrigin (OriginID: [[O_RET_VAL]])
// CHECK:   Expire (LoanID: [[L_X]])
}


// Pointer Assignment and Return
// CHECK-LABEL: Function: assign_and_return_local_addr
// CHECK-NEXT: Block B{{[0-9]+}}:
MyObj* assign_and_return_local_addr() {
  MyObj y{20};
  MyObj* ptr1 = &y;
// CHECK: Issue (LoanID: [[L_Y:[0-9]+]], OriginID: [[O_ADDR_Y:[0-9]+]])
// CHECK: AssignOrigin (DestID: [[O_PTR1:[0-9]+]], SrcID: [[O_ADDR_Y]])
  MyObj* ptr2 = ptr1;
// CHECK: AssignOrigin (DestID: [[O_PTR1_RVAL:[0-9]+]], SrcID: [[O_PTR1]])
// CHECK: AssignOrigin (DestID: [[O_PTR2:[0-9]+]], SrcID: [[O_PTR1_RVAL]])
  ptr2 = ptr1;
// CHECK: AssignOrigin (DestID: [[O_PTR1_RVAL_2:[0-9]+]], SrcID: [[O_PTR1]])
// CHECK: AssignOrigin (DestID: [[O_PTR2]], SrcID: [[O_PTR1_RVAL_2]])
  ptr2 = ptr2; // Self assignment.
// CHECK: AssignOrigin (DestID: [[O_PTR2_RVAL:[0-9]+]], SrcID: [[O_PTR2]])
// CHECK: AssignOrigin (DestID: [[O_PTR2]], SrcID: [[O_PTR2_RVAL]])
  return ptr2;
// CHECK: AssignOrigin (DestID: [[O_PTR2_RVAL_2:[0-9]+]], SrcID: [[O_PTR2]])
// CHECK: ReturnOfOrigin (OriginID: [[O_PTR2_RVAL_2]])
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
// CHECK: Issue (LoanID: [[L_OBJ:[0-9]+]], OriginID: [[O_ADDR_OBJ:[0-9]+]])
// CHECK: AssignOrigin (DestID: [[O_POBJ:[0-9]+]], SrcID: [[O_ADDR_OBJ]])
// CHECK: Expire (LoanID: [[L_OBJ]])
}


// FIXME: No expire for Trivial Destructors
// CHECK-LABEL: Function: loan_expires_trivial
// CHECK-NEXT: Block B{{[0-9]+}}:
void loan_expires_trivial() {
  int trivial_obj = 1;
  int* pTrivialObj = &trivial_obj;
// CHECK: Issue (LoanID: [[L_TRIVIAL_OBJ:[0-9]+]], OriginID: [[O_ADDR_TRIVIAL_OBJ:[0-9]+]])
// CHECK: AssignOrigin (DestID: [[O_PTOBJ:[0-9]+]], SrcID: [[O_ADDR_TRIVIAL_OBJ]])
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
  // CHECK: Issue (LoanID: [[L_A:[0-9]+]], OriginID: [[O_ADDR_A:[0-9]+]])
  // CHECK: AssignOrigin (DestID: [[O_P:[0-9]+]], SrcID: [[O_ADDR_A]])
  else
    p = &b;
  // CHECK: Issue (LoanID: [[L_B:[0-9]+]], OriginID: [[O_ADDR_B:[0-9]+]])
  // CHECK: AssignOrigin (DestID: [[O_P]], SrcID: [[O_ADDR_B]])
  int *q = p;
  // CHECK: AssignOrigin (DestID: [[O_P_RVAL:[0-9]+]], SrcID: [[O_P]])
  // CHECK: AssignOrigin (DestID: [[O_Q:[0-9]+]], SrcID: [[O_P_RVAL]])
}


// CHECK-LABEL: Function: overwrite_origin
void overwrite_origin() {
  MyObj s1;
  MyObj s2;
  MyObj* p = &s1;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue (LoanID: [[L_S1:[0-9]+]], OriginID: [[O_ADDR_S1:[0-9]+]])
// CHECK:   AssignOrigin (DestID: [[O_P:[0-9]+]], SrcID: [[O_ADDR_S1]])
  p = &s2;
// CHECK:   Issue (LoanID: [[L_S2:[0-9]+]], OriginID: [[O_ADDR_S2:[0-9]+]])
// CHECK:   AssignOrigin (DestID: [[O_P]], SrcID: [[O_ADDR_S2]])
// CHECK:   Expire (LoanID: [[L_S2]])
// CHECK:   Expire (LoanID: [[L_S1]])
}


// CHECK-LABEL: Function: reassign_to_null
void reassign_to_null() {
  MyObj s1;
  MyObj* p = &s1;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue (LoanID: [[L_S1:[0-9]+]], OriginID: [[O_ADDR_S1:[0-9]+]])
// CHECK:   AssignOrigin (DestID: [[O_P:[0-9]+]], SrcID: [[O_ADDR_S1]])
  p = nullptr;
// CHECK:   AssignOrigin (DestID: [[O_P]], SrcID: [[O_NULLPTR:[0-9]+]])
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
// CHECK:   Issue (LoanID: [[L_S1:[0-9]+]], OriginID: [[O_ADDR_S1:[0-9]+]])
// CHECK:   AssignOrigin (DestID: [[O_P:[0-9]+]], SrcID: [[O_ADDR_S1]])
  if (condition) {
    p = &s2;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Issue (LoanID: [[L_S2:[0-9]+]], OriginID: [[O_ADDR_S2:[0-9]+]])
// CHECK:   AssignOrigin (DestID: [[O_P]], SrcID: [[O_ADDR_S2]])
  }
// CHECK: Block B{{[0-9]+}}:
// CHECK:   Expire (LoanID: [[L_S2]])
// CHECK:   Expire (LoanID: [[L_S1]])
}


// CHECK-LABEL: Function: nested_scopes
void nested_scopes() {
  MyObj* p = nullptr;
// CHECK: Block B{{[0-9]+}}:
// CHECK:   AssignOrigin (DestID: [[O_NULLPTR_CAST:[0-9]+]], SrcID: [[O_NULLPTR:[0-9]+]])
// CHECK:   AssignOrigin (DestID: [[O_P:[0-9]+]], SrcID: [[O_NULLPTR_CAST]])
  {
    MyObj outer;
    p = &outer;
// CHECK:   Issue (LoanID: [[L_OUTER:[0-9]+]], OriginID: [[O_ADDR_OUTER:[0-9]+]])
// CHECK:   AssignOrigin (DestID: [[O_P]], SrcID: [[O_ADDR_OUTER]])
    {
      MyObj inner;
      p = &inner;
// CHECK:   Issue (LoanID: [[L_INNER:[0-9]+]], OriginID: [[O_ADDR_INNER:[0-9]+]])
// CHECK:   AssignOrigin (DestID: [[O_P]], SrcID: [[O_ADDR_INNER]])
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
// CHECK:   Issue (LoanID: [[L_A:[0-9]+]], OriginID: [[O_ADDR_A:[0-9]+]])
// CHECK:   AssignOrigin (DestID: [[O_P:[0-9]+]], SrcID: [[O_ADDR_A]])
  int **pp = &p;
// CHECK:   Issue (LoanID: [[L_P:[0-9]+]], OriginID: [[O_ADDR_P:[0-9]+]])
// CHECK:   AssignOrigin (DestID: [[O_PP:[0-9]+]], SrcID: [[O_ADDR_P]])

// FIXME: The Origin for the RHS is broken
  int *q = *pp;
// CHECK:   AssignOrigin (DestID: [[O_Q:[0-9]+]], SrcID: {{[0-9]+}})
}
