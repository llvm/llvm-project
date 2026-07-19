// RUN: %clang_cc1 -ast-dump -ast-dump-filter bar -fbounds-safety %s | FileCheck %s --check-prefix=FBOUNDS
// RUN: %clang_cc1 -ast-dump -ast-dump-filter bar -fexperimental-bounds-safety-attributes %s | FileCheck %s --check-prefix=FATTR_ONLY

#include <ptrcheck.h>

typedef int * __single _Nonnull baz_t;

void bar(int len, baz_t __counted_by(len) p);

// FBOUNDS:      FunctionDecl {{.*}} bar 'void (int, int *__single __counted_by(len) _Nonnull)'
// FBOUNDS-NEXT: ParmVarDecl {{.*}} len 'int'
// FBOUNDS-NEXT: DependerDeclsAttr
// FBOUNDS-NEXT: ParmVarDecl {{.*}} p 'int *__single __counted_by(len) _Nonnull':'int *__single'

// FATTR_ONLY:      FunctionDecl {{.*}} bar 'void (int, int * __counted_by(len) _Nonnull__single)'
// FATTR_ONLY-NEXT: ParmVarDecl {{.*}} len 'int'
// FATTR_ONLY-NEXT: DependerDeclsAttr
// FATTR_ONLY-NEXT: ParmVarDecl {{.*}} p 'int * __counted_by(len) _Nonnull__single':'int *'
