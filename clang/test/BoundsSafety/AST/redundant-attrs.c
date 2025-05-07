
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s | FileCheck %s
#include <ptrcheck.h>

typedef int *__bidi_indexable bidiPtr;
bidiPtr __bidi_indexable ptrBoundBound;

#define bidiPtr2 int *__bidi_indexable
bidiPtr2 __bidi_indexable ptrBoundBound2;

// CHECK:      TypedefDecl {{.*}} referenced bidiPtr 'int *__bidi_indexable'
// CHECK-NEXT:   PointerType {{.*}} 'int *__bidi_indexable'
// CHECK-NEXT:     BuiltinType {{.*}} 'int'
// CHECK:      VarDecl {{.*}} ptrBoundBound 'bidiPtr':'int *__bidi_indexable'
// CHECK-NEXT: VarDecl {{.*}} ptrBoundBound2 'int *__bidi_indexable'

typedef const int * _Nullable __bidi_indexable my_c_ptr_nullable_bidi_t;
my_c_ptr_nullable_bidi_t __bidi_indexable def_c_nullable_bidi_ptr;
// CHECK: TypedefDecl {{.*}} referenced my_c_ptr_nullable_bidi_t 'const int *__bidi_indexable _Nullable':'const int *__bidi_indexable'
// CHECK-NEXT: AttributedType {{.*}} 'const int *__bidi_indexable _Nullable' sugar
// CHECK-NEXT: PointerType {{.*}} 'const int *__bidi_indexable'
// CHECK-NEXT: QualType {{.*}} 'const int' const
// CHECK-NEXT: BuiltinType {{.*}} 'int'
// CHECK-NEXT: VarDecl {{.*}} def_c_nullable_bidi_ptr 'const int *__bidi_indexable _Nullable':'const int *__bidi_indexable'
