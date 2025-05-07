
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// CHECK: VarDecl {{.+}} array1 'int[__terminated_by(0) 3]':'int[3]'
int array1[__null_terminated 3] = {1, 2, 0};

// CHECK: VarDecl {{.+}} array2 'int[__terminated_by(42) 3]':'int[3]'
int array2[__terminated_by(42) 3] = {1, 2, 42};

// CHECK: VarDecl {{.+}} incomplete_array1 'int[__terminated_by(0) 3]':'int[3]'
int incomplete_array1[__null_terminated] = {1, 2, 0};

// CHECK: VarDecl {{.+}} incomplete_array2 'int[__terminated_by(42) 3]':'int[3]'
int incomplete_array2[__terminated_by(42)] = {1, 2, 42};

// CHECK: VarDecl {{.+}} ptr1 'int *__single __terminated_by(0)':'int *__single'
int *__null_terminated ptr1 = array1;

// CHECK: VarDecl {{.+}} ptr2 'int *__single __terminated_by(42)':'int *__single'
int *__terminated_by(42) ptr2 = array2;

// CHECK: VarDecl {{.+}} ptr_array1 'int *__single __terminated_by(42)[__terminated_by(0) 3]':'int *__single __terminated_by(42)[3]'
int *__terminated_by(42) ptr_array1[__null_terminated 3] = {array2, array2, 0};

// CHECK: VarDecl {{.+}} ptr_ptr1 'int *__single __terminated_by(42)*__single __terminated_by(0)':'int *__single __terminated_by(42)*__single'
int *__terminated_by(42) *__null_terminated ptr_ptr1 = ptr_array1;

// CHECK:      RecordDecl {{.+}} foo
// CHECK-NEXT: FieldDecl {{.+}} array3 'int[__terminated_by(0) 2]':'int[2]'
// CHECK-NEXT: FieldDecl {{.+}} ptr3 'int *__single __terminated_by(0)':'int *__single'
struct foo {
  int array3[__null_terminated 2];
  int *__null_terminated ptr3;
};

// CHECK: ParmVarDecl {{.+}} ptr4 'int *__single __terminated_by(0)':'int *__single'
void foo(int *__null_terminated ptr4);

// CHECK: ParmVarDecl {{.+}} ptr_cv_nt 'int *__single __terminated_by(0)const volatile':'int *__singleconst volatile'
void quals_cv_nt(int *const volatile __null_terminated ptr_cv_nt);

// CHECK: ParmVarDecl {{.+}} ptr_nt_cv 'int *__single __terminated_by(0)const volatile':'int *__singleconst volatile'
void quals_nt_cv(int *__null_terminated const volatile ptr_nt_cv);

// CHECK: ParmVarDecl {{.+}} nested_ptr_cv_nt 'int *__single __terminated_by(0)const volatile *__single __terminated_by(0)const volatile':'int *__single __terminated_by(0)const volatile *__singleconst volatile'
void nested_quals_cv_nt(int *__null_terminated const volatile *const volatile __null_terminated nested_ptr_cv_nt);

// CHECK: ParmVarDecl {{.+}} nested_ptr_nt_cv 'int *__single __terminated_by(0)const volatile *__single __terminated_by(0)const volatile':'int *__single __terminated_by(0)const volatile *__singleconst volatile'
void nested_quals_nt_cv(int *__null_terminated const volatile *__null_terminated const volatile nested_ptr_nt_cv);

// CHECK: TypedefDecl {{.+}} my_int_t 'int *'
typedef int *my_int_t;

// CHECK: ParmVarDecl {{.+}} typedef_nt 'int *__single __terminated_by(0)':'int *__single'
void typedef_nt(my_int_t __null_terminated typedef_nt);

// CHECK: ParmVarDecl {{.+}} typedef_nt_c 'int *__single __terminated_by(0)const':'int *__singleconst'
void typedef_nt_c(my_int_t __null_terminated const typedef_nt_c);

// CHECK: ParmVarDecl {{.+}} typedef_c_nt 'int *__single __terminated_by(0)const':'int *__singleconst'
void typedef_c_nt(my_int_t const __null_terminated typedef_c_nt);

// CHECK: TypedefDecl {{.+}} my_cint_t 'int *const'
typedef int *const my_cint_t;

// CHECK: ParmVarDecl {{.+}} ctypedef_nt 'int *__single __terminated_by(0)const':'int *__singleconst'
void ctypedef_nt(my_cint_t __null_terminated ctypedef_nt);

// CHECK: ParmVarDecl {{.+}} ctypedef_nt_c 'int *__single __terminated_by(0)const':'int *__singleconst'
void ctypedef_nt_c(my_cint_t __null_terminated const ctypedef_nt_c);

// CHECK: ParmVarDecl {{.+}} ctypedef_c_nt 'int *__single __terminated_by(0)const':'int *__singleconst'
void ctypedef_c_nt(my_cint_t const __null_terminated ctypedef_c_nt);

#define my_ptr_c_nt_t int *const __null_terminated
// CHECK: VarDecl {{.+}} def_c_nt_nt 'int *__single __terminated_by(0)const':'int *__singleconst'
my_ptr_c_nt_t __null_terminated def_c_nt_nt;

#define my_ptr_nt_nullable_t int *__null_terminated _Nullable
// CHECK: VarDecl {{.+}} def_nt_nullable_nt 'int *__single __terminated_by(0) _Nullable':'int *__single'
my_ptr_nt_nullable_t __null_terminated def_nt_nullable_nt;

#define my_ptr_nullable_nt_t int *_Nullable __null_terminated
// CHECK: VarDecl {{.+}} def_nullable_nt_nt 'int *__single __terminated_by(0) _Nullable':'int *__single'
my_ptr_nullable_nt_t __null_terminated def_nullable_nt_nt;

#define my_c_a_int_t const int __attribute__((aligned(64)))
typedef my_c_a_int_t * __attribute__((align_value(64))) _Nullable __null_terminated my_c_ptr_nullable_nt_t;
// CHECK: TypedefDecl {{.*}} referenced my_c_ptr_nullable_nt_t 'const int * __terminated_by(0) _Nullable':'const int *'
// CHECK-NEXT: |-AttributedType {{.*}} 'const int * __terminated_by(0) _Nullable' sugar
// CHECK-NEXT: | `-ValueTerminatedType {{.*}} 'const int * __terminated_by(0)' sugar
// CHECK-NEXT: |   `-PointerType {{.*}} 'const int *'
// CHECK-NEXT: |     `-QualType {{.*}} 'const int' const
// CHECK-NEXT: |       `-BuiltinType {{.*}} 'int'
// CHECK-NEXT: |-AlignedAttr {{.*}} aligned
// CHECK-NEXT: | `-ConstantExpr {{.*}} 'int'
// CHECK-NEXT: |   |-value: Int 64
// CHECK-NEXT: |   `-IntegerLiteral {{.*}} 'int' 64
// CHECK-NEXT: `-AlignValueAttr {{.*}}
// CHECK-NEXT:   `-ConstantExpr {{.*}} 'int'
// CHECK-NEXT:     |-value: Int 64
// CHECK-NEXT:     `-IntegerLiteral {{.*}} 'int' 64
// CHECK: VarDecl {{.*}} def_c_nullable_nt_nt 'const int *__single __terminated_by(0) _Nullable':'const int *__single'
my_c_ptr_nullable_nt_t __null_terminated def_c_nullable_nt_nt;

