
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

typedef void cb_t(int *__counted_by(count), int count);
typedef void sb_t(void *__sized_by(size), int size);
typedef void eb_t(void *__ended_by(end) start, void *end);

typedef void (*cb_ptr_t)(int *__counted_by(count), int count);
typedef void (*sb_ptr_t)(void *__sized_by(size), int size);
typedef void (*eb_ptr_t)(void *__ended_by(end) start, void *end);

cb_ptr_t g_cb_ptr;
sb_ptr_t g_sb_ptr;
eb_ptr_t g_eb_ptr;

cb_t *g_ptr_cb;
sb_t *g_ptr_sb;
eb_t *g_ptr_eb;

void foo(
    cb_ptr_t a_cb_ptr,
    sb_ptr_t a_sb_ptr,
    eb_ptr_t a_eb_ptr,
    cb_t *a_ptr_cb,
    sb_t *a_ptr_sb,
    eb_t *a_ptr_eb) {
  cb_ptr_t l_cb_ptr;
  sb_ptr_t l_sb_ptr;
  eb_ptr_t l_eb_ptr;

  cb_t *l_ptr_cb;
  sb_t *l_ptr_sb;
  eb_t *l_ptr_eb;
}

// CHECK: |-TypedefDecl {{.+}} cb_t 'void (int *__single __counted_by(count), int)'
// CHECK: | `-FunctionProtoType {{.+}} 'void (int *__single __counted_by(count), int)'
// CHECK: |   |-BuiltinType {{.+}} 'void'
// CHECK: |   |-CountAttributedType {{.+}} 'int *__single __counted_by(count)' sugar
// CHECK: |   | `-PointerType {{.+}} 'int *__single'
// CHECK: |   |   `-BuiltinType {{.+}} 'int'
// CHECK: |   `-BuiltinType {{.+}} 'int'
// CHECK: |-TypedefDecl {{.+}} sb_t 'void (void *__single __sized_by(size), int)'
// CHECK: | `-FunctionProtoType {{.+}} 'void (void *__single __sized_by(size), int)'
// CHECK: |   |-BuiltinType {{.+}} 'void'
// CHECK: |   |-CountAttributedType {{.+}} 'void *__single __sized_by(size)' sugar
// CHECK: |   | `-PointerType {{.+}} 'void *__single'
// CHECK: |   |   `-BuiltinType {{.+}} 'void'
// CHECK: |   `-BuiltinType {{.+}} 'int'
// CHECK: |-TypedefDecl {{.+}} eb_t 'void (void *__single __ended_by(end), void *__single /* __started_by(start) */ )'
// CHECK: | `-FunctionProtoType {{.+}} 'void (void *__single __ended_by(end), void *__single /* __started_by(start) */ )'
// CHECK: |   |-BuiltinType {{.+}} 'void'
// CHECK: |   |-DynamicRangePointerType {{.+}} 'void *__single __ended_by(end)' sugar
// CHECK: |   | `-PointerType {{.+}} 'void *__single'
// CHECK: |   |   `-BuiltinType {{.+}} 'void'
// CHECK: |   `-DynamicRangePointerType {{.+}} 'void *__single /* __started_by(start) */ ' sugar
// CHECK: |     `-PointerType {{.+}} 'void *__single'
// CHECK: |       `-BuiltinType {{.+}} 'void'
// CHECK: |-TypedefDecl {{.+}} cb_ptr_t 'void (*)(int *__single __counted_by(count), int)'
// CHECK: | `-PointerType {{.+}} 'void (*)(int *__single __counted_by(count), int)'
// CHECK: |   `-ParenType {{.+}} 'void (int *__single __counted_by(count), int)' sugar
// CHECK: |     `-FunctionProtoType {{.+}} 'void (int *__single __counted_by(count), int)'
// CHECK: |       |-BuiltinType {{.+}} 'void'
// CHECK: |       |-CountAttributedType {{.+}} 'int *__single __counted_by(count)' sugar
// CHECK: |       | `-PointerType {{.+}} 'int *__single'
// CHECK: |       |   `-BuiltinType {{.+}} 'int'
// CHECK: |       `-BuiltinType {{.+}} 'int'
// CHECK: |-TypedefDecl {{.+}} sb_ptr_t 'void (*)(void *__single __sized_by(size), int)'
// CHECK: | `-PointerType {{.+}} 'void (*)(void *__single __sized_by(size), int)'
// CHECK: |   `-ParenType {{.+}} 'void (void *__single __sized_by(size), int)' sugar
// CHECK: |     `-FunctionProtoType {{.+}} 'void (void *__single __sized_by(size), int)'
// CHECK: |       |-BuiltinType {{.+}} 'void'
// CHECK: |       |-CountAttributedType {{.+}} 'void *__single __sized_by(size)' sugar
// CHECK: |       | `-PointerType {{.+}} 'void *__single'
// CHECK: |       |   `-BuiltinType {{.+}} 'void'
// CHECK: |       `-BuiltinType {{.+}} 'int'
// CHECK: |-TypedefDecl {{.+}} eb_ptr_t 'void (*)(void *__single __ended_by(end), void *__single /* __started_by(start) */ )'
// CHECK: | `-PointerType {{.+}} 'void (*)(void *__single __ended_by(end), void *__single /* __started_by(start) */ )'
// CHECK: |   `-ParenType {{.+}} 'void (void *__single __ended_by(end), void *__single /* __started_by(start) */ )' sugar
// CHECK: |     `-FunctionProtoType {{.+}} 'void (void *__single __ended_by(end), void *__single /* __started_by(start) */ )'
// CHECK: |       |-BuiltinType {{.+}} 'void'
// CHECK: |       |-DynamicRangePointerType {{.+}} 'void *__single __ended_by(end)' sugar
// CHECK: |       | `-PointerType {{.+}} 'void *__single'
// CHECK: |       |   `-BuiltinType {{.+}} 'void'
// CHECK: |       `-DynamicRangePointerType {{.+}} 'void *__single /* __started_by(start) */ ' sugar
// CHECK: |         `-PointerType {{.+}} 'void *__single'
// CHECK: |           `-BuiltinType {{.+}} 'void'
// CHECK: |-VarDecl {{.+}} g_cb_ptr 'void (*__single)(int *__single __counted_by(count), int)'
// CHECK: |-VarDecl {{.+}} g_sb_ptr 'void (*__single)(void *__single __sized_by(size), int)'
// CHECK: |-VarDecl {{.+}} g_eb_ptr 'void (*__single)(void *__single __ended_by(end), void *__single /* __started_by(start) */ )'
// CHECK: |-VarDecl {{.+}} g_ptr_cb 'cb_t *__single'
// CHECK: |-VarDecl {{.+}} g_ptr_sb 'sb_t *__single'
// CHECK: |-VarDecl {{.+}} g_ptr_eb 'eb_t *__single'
// CHECK: `-FunctionDecl {{.+}} foo 'void (void (*__single)(int *__single __counted_by(count), int), void (*__single)(void *__single __sized_by(size), int), void (*__single)(void *__single __ended_by(end), void *__single /* __started_by(start) */ ), cb_t *__single, sb_t *__single, eb_t *__single)'
// CHECK:   |-ParmVarDecl {{.+}} a_cb_ptr 'void (*__single)(int *__single __counted_by(count), int)'
// CHECK:   |-ParmVarDecl {{.+}} a_sb_ptr 'void (*__single)(void *__single __sized_by(size), int)'
// CHECK:   |-ParmVarDecl {{.+}} a_eb_ptr 'void (*__single)(void *__single __ended_by(end), void *__single /* __started_by(start) */ )'
// CHECK:   |-ParmVarDecl {{.+}} a_ptr_cb 'cb_t *__single'
// CHECK:   |-ParmVarDecl {{.+}} a_ptr_sb 'sb_t *__single'
// CHECK:   |-ParmVarDecl {{.+}} a_ptr_eb 'eb_t *__single'
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} l_cb_ptr 'void (*__single)(int *__single __counted_by(count), int)'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} l_sb_ptr 'void (*__single)(void *__single __sized_by(size), int)'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} l_eb_ptr 'void (*__single)(void *__single __ended_by(end), void *__single /* __started_by(start) */ )'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} l_ptr_cb 'cb_t *__single'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.+}} l_ptr_sb 'sb_t *__single'
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl {{.+}} l_ptr_eb 'eb_t *__single'
