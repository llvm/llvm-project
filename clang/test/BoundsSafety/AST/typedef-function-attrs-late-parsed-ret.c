
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

typedef int *__counted_by(count) cb_t(int count);
typedef void *__sized_by(size) sb_t(int size);
typedef int *__ended_by(end) eb_t(int *end);

typedef int *__counted_by(count) (*cb_ptr_t)(int count);
typedef void *__sized_by(size) (*sb_ptr_t)(int size);
typedef int *__ended_by(end) (*eb_ptr_t)(int *end);

cb_t g_cb;
sb_t g_sb;
eb_t g_eb;

cb_ptr_t g_cb_ptr;
sb_ptr_t g_sb_ptr;
eb_ptr_t g_eb_ptr;

cb_t *g_ptr_cb;
sb_t *g_ptr_sb;
eb_t *g_ptr_eb;

void foo(
    cb_t a_cb,
    sb_t a_sb,
    eb_t a_eb,
    cb_ptr_t a_cb_ptr,
    sb_ptr_t a_sb_ptr,
    eb_ptr_t a_eb_ptr,
    cb_t *a_ptr_cb,
    sb_t *a_ptr_sb,
    eb_t *a_ptr_eb) {
  cb_t l_cb;
  sb_t l_sb;
  eb_t l_eb;

  cb_ptr_t l_cb_ptr;
  sb_ptr_t l_sb_ptr;
  eb_ptr_t l_eb_ptr;

  cb_t *l_ptr_cb;
  sb_t *l_ptr_sb;
  eb_t *l_ptr_eb;
}

// CHECK: |-TypedefDecl {{.+}} referenced cb_t 'int *__single __counted_by(count)(int)'
// CHECK: | `-FunctionProtoType {{.+}} 'int *__single __counted_by(count)(int)' cdecl
// CHECK: |   |-CountAttributedType {{.+}} 'int *__single __counted_by(count)' sugar
// CHECK: |   | `-PointerType {{.+}} 'int *__single'
// CHECK: |   |   `-BuiltinType {{.+}} 'int'
// CHECK: |   `-BuiltinType {{.+}} 'int'
// CHECK: |-TypedefDecl {{.+}} referenced sb_t 'void *__single __sized_by(size)(int)'
// CHECK: | `-FunctionProtoType {{.+}} 'void *__single __sized_by(size)(int)' cdecl
// CHECK: |   |-CountAttributedType {{.+}} 'void *__single __sized_by(size)' sugar
// CHECK: |   | `-PointerType {{.+}} 'void *__single'
// CHECK: |   |   `-BuiltinType {{.+}} 'void'
// CHECK: |   `-BuiltinType {{.+}} 'int'
// CHECK: |-TypedefDecl {{.+}} referenced eb_t 'int *__single __ended_by(end)(int *__single)'
// CHECK: | `-FunctionProtoType {{.+}} 'int *__single __ended_by(end)(int *__single)' cdecl
// CHECK: |   |-DynamicRangePointerType {{.+}} 'int *__single __ended_by(end)' sugar
// CHECK: |   | `-PointerType {{.+}} 'int *__single'
// CHECK: |   |   `-BuiltinType {{.+}} 'int'
// CHECK: |   `-AttributedType {{.+}} 'int *__single' sugar
// CHECK: |     `-PointerType {{.+}} 'int *__single'
// CHECK: |       `-BuiltinType {{.+}} 'int'
// CHECK: |-TypedefDecl {{.+}} referenced cb_ptr_t 'int *__single __counted_by(count)(*)(int)'
// CHECK: | `-PointerType {{.+}} 'int *__single __counted_by(count)(*)(int)'
// CHECK: |   `-ParenType {{.+}} 'int *__single __counted_by(count)(int)' sugar
// CHECK: |     `-FunctionProtoType {{.+}} 'int *__single __counted_by(count)(int)' cdecl
// CHECK: |       |-CountAttributedType {{.+}} 'int *__single __counted_by(count)' sugar
// CHECK: |       | `-PointerType {{.+}} 'int *__single'
// CHECK: |       |   `-BuiltinType {{.+}} 'int'
// CHECK: |       `-BuiltinType {{.+}} 'int'
// CHECK: |-TypedefDecl {{.+}} referenced sb_ptr_t 'void *__single __sized_by(size)(*)(int)'
// CHECK: | `-PointerType {{.+}} 'void *__single __sized_by(size)(*)(int)'
// CHECK: |   `-ParenType {{.+}} 'void *__single __sized_by(size)(int)' sugar
// CHECK: |     `-FunctionProtoType {{.+}} 'void *__single __sized_by(size)(int)' cdecl
// CHECK: |       |-CountAttributedType {{.+}} 'void *__single __sized_by(size)' sugar
// CHECK: |       | `-PointerType {{.+}} 'void *__single'
// CHECK: |       |   `-BuiltinType {{.+}} 'void'
// CHECK: |       `-BuiltinType {{.+}} 'int'
// CHECK: |-TypedefDecl {{.+}} referenced eb_ptr_t 'int *__single __ended_by(end)(*)(int *__single)'
// CHECK: | `-PointerType {{.+}} 'int *__single __ended_by(end)(*)(int *__single)'
// CHECK: |   `-ParenType {{.+}} 'int *__single __ended_by(end)(int *__single)' sugar
// CHECK: |     `-FunctionProtoType {{.+}} 'int *__single __ended_by(end)(int *__single)' cdecl
// CHECK: |       |-DynamicRangePointerType {{.+}} 'int *__single __ended_by(end)' sugar
// CHECK: |       | `-PointerType {{.+}} 'int *__single'
// CHECK: |       |   `-BuiltinType {{.+}} 'int'
// CHECK: |       `-AttributedType {{.+}} 'int *__single' sugar
// CHECK: |         `-PointerType {{.+}} 'int *__single'
// CHECK: |           `-BuiltinType {{.+}} 'int'
// CHECK: |-FunctionDecl {{.+}} g_cb 'int *__single __counted_by(count)(int)'
// CHECK: | `-ParmVarDecl {{.+}} implicit used count 'int'
// CHECK: |-FunctionDecl {{.+}} g_sb 'void *__single __sized_by(size)(int)'
// CHECK: | `-ParmVarDecl {{.+}} implicit used size 'int'
// CHECK: |-FunctionDecl {{.+}} g_eb 'int *__single __ended_by(end)(int *__single)'
// CHECK: | `-ParmVarDecl {{.+}} implicit used end 'int *__single'
// CHECK: |-VarDecl {{.+}} g_cb_ptr 'int *__single __counted_by(count)(*__single)(int)'
// CHECK: |-VarDecl {{.+}} g_sb_ptr 'void *__single __sized_by(size)(*__single)(int)'
// CHECK: |-VarDecl {{.+}} g_eb_ptr 'int *__single __ended_by(end)(*__single)(int *__single)'
// CHECK: |-VarDecl {{.+}} g_ptr_cb 'cb_t *__single'
// CHECK: |-VarDecl {{.+}} g_ptr_sb 'sb_t *__single'
// CHECK: |-VarDecl {{.+}} g_ptr_eb 'eb_t *__single'
// CHECK: `-FunctionDecl {{.+}} foo 'void (cb_t *__single, sb_t *__single, eb_t *__single, int *__single __counted_by(count)(*__single)(int), void *__single __sized_by(size)(*__single)(int), int *__single __ended_by(end)(*__single)(int *__single), cb_t *__single, sb_t *__single, eb_t *__single)'
// CHECK:   |-ParmVarDecl {{.+}} a_cb 'cb_t *__single'
// CHECK:   |-ParmVarDecl {{.+}} a_sb 'sb_t *__single'
// CHECK:   |-ParmVarDecl {{.+}} a_eb 'eb_t *__single'
// CHECK:   |-ParmVarDecl {{.+}} a_cb_ptr 'int *__single __counted_by(count)(*__single)(int)'
// CHECK:   |-ParmVarDecl {{.+}} a_sb_ptr 'void *__single __sized_by(size)(*__single)(int)'
// CHECK:   |-ParmVarDecl {{.+}} a_eb_ptr 'int *__single __ended_by(end)(*__single)(int *__single)'
// CHECK:   |-ParmVarDecl {{.+}} a_ptr_cb 'cb_t *__single'
// CHECK:   |-ParmVarDecl {{.+}} a_ptr_sb 'sb_t *__single'
// CHECK:   |-ParmVarDecl {{.+}} a_ptr_eb 'eb_t *__single'
// CHECK:   `-CompoundStmt {{.+}}
// CHECK:     |-DeclStmt
// CHECK:     | `-FunctionDecl {{.+}} l_cb 'int *__single __counted_by(count)(int)'
// CHECK:     |   `-ParmVarDecl {{.+}} implicit used count 'int'
// CHECK:     |-DeclStmt
// CHECK:     | `-FunctionDecl {{.+}} l_sb 'void *__single __sized_by(size)(int)'
// CHECK:     |   `-ParmVarDecl {{.+}} implicit used size 'int'
// CHECK:     |-DeclStmt
// CHECK:     | `-FunctionDecl {{.+}} l_eb 'int *__single __ended_by(end)(int *__single)'
// CHECK:     |   `-ParmVarDecl {{.+}} implicit used end 'int *__single'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} l_cb_ptr 'int *__single __counted_by(count)(*__single)(int)'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} l_sb_ptr 'void *__single __sized_by(size)(*__single)(int)'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} l_eb_ptr 'int *__single __ended_by(end)(*__single)(int *__single)'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} l_ptr_cb 'cb_t *__single'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} l_ptr_sb 'sb_t *__single'
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl {{.+}} l_ptr_eb 'eb_t *__single'
