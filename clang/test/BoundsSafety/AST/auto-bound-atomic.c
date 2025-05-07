
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

struct field {
  // CHECK: FieldDecl {{.+}} f1 '_Atomic(int *__single)'
  int *_Atomic f1;
  // CHECK: FieldDecl {{.+}} f2 '_Atomic(int *__single)'
  _Atomic(int *) f2;
};

// CHECK: FunctionDecl {{.+}} ret1 '_Atomic(int *__single) (void)'
int *_Atomic ret1(void);
// CHECK: FunctionDecl {{.+}} ret2 '_Atomic(int *__single) (void)'
_Atomic(int *) ret2(void);

// CHECK: ParmVarDecl {{.+}} p1 '_Atomic(int *__single)'
void parm1(int *_Atomic p1);
// CHECK: ParmVarDecl {{.+}} p2 '_Atomic(int *__single)'
void parm2(_Atomic(int *) p2);

void locals(void) {
  // CHECK: VarDecl {{.+}} l1 '_Atomic(_Atomic(int *__single) *__single)'
  int *_Atomic *_Atomic __single l1;
  // CHECK: VarDecl {{.+}} l2 '_Atomic(_Atomic(int *__single) *__single)'
  _Atomic(int *) *_Atomic __single l2;

  // CHECK: VarDecl {{.+}} l3 '_Atomic(_Atomic(int *__single) *__unsafe_indexable)'
  int *_Atomic *_Atomic __unsafe_indexable l3;
  // CHECK: VarDecl {{.+}} l4 '_Atomic(_Atomic(int *__single) *__unsafe_indexable)'
  _Atomic(int *) *_Atomic __unsafe_indexable l4;
}
