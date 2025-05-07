
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

typedef int *my_ptr_t;
typedef my_ptr_t const my_cptr_t;

typedef int *__indexable my_iptr_t;
typedef my_iptr_t const my_ciptr_t;

void foo(void) {
  my_ptr_t bp1;
  my_cptr_t bp2;
  const my_ptr_t bp3;

  my_iptr_t ip1;
  my_ciptr_t ip2;
  const my_iptr_t ip3;
}

// CHECK: |-TypedefDecl {{.*}} referenced my_ptr_t 'int *'
// CHECK: | `-PointerType {{.*}} 'int *'
// CHECK: |   `-BuiltinType {{.*}} 'int'
// CHECK: |-TypedefDecl {{.*}} referenced my_cptr_t 'const my_ptr_t':'int *const'
// CHECK: | `-QualType {{.*}} 'const my_ptr_t' const
// CHECK: |   `-TypedefType {{.*}} 'my_ptr_t' sugar
// CHECK: |     |-Typedef {{.*}} 'my_ptr_t'
// CHECK: |     `-PointerType {{.*}} 'int *'
// CHECK: |       `-BuiltinType {{.*}} 'int'
// CHECK: |-TypedefDecl {{.*}} referenced my_iptr_t 'int *__indexable'
// CHECK: | `-PointerType {{.*}} 'int *__indexable'
// CHECK: |   `-BuiltinType {{.*}} 'int'
// CHECK: |-TypedefDecl {{.*}} referenced my_ciptr_t 'const my_iptr_t':'int *__indexableconst'
// CHECK: | `-QualType {{.*}} 'const my_iptr_t' const
// CHECK: |   `-TypedefType {{.*}} 'my_iptr_t' sugar
// CHECK: |     |-Typedef {{.*}} 'my_iptr_t'
// CHECK: |     `-PointerType {{.*}} 'int *__indexable'
// CHECK: |       `-BuiltinType {{.*}} 'int'
// CHECK: `-FunctionDecl {{.*}} foo 'void (void)'
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} bp1 'int *__bidi_indexable'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} bp2 'int *__bidi_indexableconst'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} bp3 'int *__bidi_indexableconst'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} ip1 'my_iptr_t':'int *__indexable'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} ip2 'my_ciptr_t':'int *__indexableconst'
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl {{.*}} ip3 'const my_iptr_t':'int *__indexableconst'
