
#include <va-list-sys.h>

// RUN: %clang_cc1 -triple arm64-apple-macosx -ast-dump -fbounds-safety %s -I %S/include | FileCheck %s --implicit-check-not "GetBoundExpr {{.+}} 'char *__single'" --implicit-check-not "GetBoundExpr {{.+}} 'char *'"
// RUN: %clang_cc1 -triple arm64-apple-macosx -ast-dump -fbounds-safety %s -I %S/include -x objective-c -fexperimental-bounds-safety-objc | FileCheck %s --implicit-check-not "GetBoundExpr {{.+}} 'char *__single'" --implicit-check-not "GetBoundExpr {{.+}} 'char *'"
extern variable_length_function func_ptr;
typedef void * (*variable_length_function2)(va_list args);
extern variable_length_function2 func_ptr2;

void func(char *dst_str, char *src_str, int len) {
  call_func(func_ptr, dst_str, src_str, len);
  call_func(func_ptr2, dst_str, src_str, len);
}

// CHECK: |-FunctionDecl [[func_static:0x[^ ]+]] {{.+}} static
// CHECK: | |-ParmVarDecl [[var_f:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_args:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-CallExpr
// CHECK: |       |-ImplicitCastExpr {{.+}} 'void *(*)(va_list)' <LValueToRValue>
// CHECK: |       | `-DeclRefExpr {{.+}} [[var_f]]
// CHECK: |       `-ImplicitCastExpr {{.+}} 'va_list':'char *' <LValueToRValue>
// CHECK: |         `-DeclRefExpr {{.+}} [[var_args]]
// CHECK: |-FunctionDecl [[func_static_1:0x[^ ]+]] {{.+}} static
// CHECK: | |-ParmVarDecl [[var_f_1:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_ap:0x[^ ]+]]
// CHECK: |   |-CallExpr
// CHECK: |   | |-ImplicitCastExpr {{.+}} 'void (*)(__builtin_va_list &, ...)' <BuiltinFnToFnPtr>
// CHECK: |   | | `-DeclRefExpr {{.+}}
// CHECK: |   | |-DeclRefExpr {{.+}} [[var_ap]]
// CHECK: |   | `-DeclRefExpr {{.+}} [[var_f_1]]
// CHECK: |   `-ReturnStmt
// CHECK: |     `-CallExpr
// CHECK: |       |-ImplicitCastExpr {{.+}} 'void *(*__single)(void *(*)(va_list), va_list)' <FunctionToPointerDecay>
// CHECK: |       | `-DeclRefExpr {{.+}} [[func_static]]
// CHECK: |       |-ImplicitCastExpr {{.+}} 'void *(*)(va_list)' <LValueToRValue>
// CHECK: |       | `-DeclRefExpr {{.+}} [[var_f_1]]
// CHECK: |       `-ImplicitCastExpr {{.+}} 'va_list':'char *' <LValueToRValue>
// CHECK: |         `-DeclRefExpr {{.+}} [[var_ap]]
// CHECK: |-VarDecl [[var_func_ptr:0x[^ ]+]]
// CHECK: |-TypedefDecl
// CHECK: | `-PointerType
// CHECK: |   `-ParenType
// CHECK: |     `-FunctionProtoType
// CHECK: |       |-PointerType
// CHECK: |       | `-BuiltinType
// CHECK: |       `-ElaboratedType
// CHECK: |         `-TypedefType
// CHECK: |           |-Typedef
// CHECK: |           `-ElaboratedType
// CHECK: |             `-TypedefType
// CHECK: |               |-Typedef
// CHECK: |               `-PointerType
// CHECK: |                 `-BuiltinType
// CHECK: |-VarDecl [[var_func_ptr2:0x[^ ]+]]
// CHECK: `-FunctionDecl [[func_func:0x[^ ]+]] {{.+}} func
// CHECK:   |-ParmVarDecl [[var_dst_str:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_src_str:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK:   `-CompoundStmt
// CHECK:     |-CallExpr
// CHECK:     | |-ImplicitCastExpr {{.+}} 'void *(*__single)(void *(*)(va_list), ...)' <FunctionToPointerDecay>
// CHECK:     | | `-DeclRefExpr {{.+}} [[func_static_1]]
// CHECK:     | |-ImplicitCastExpr {{.+}} 'void *(*)(va_list)' <BoundsSafetyPointerCast>
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'void *(*__single)(char *)' <BitCast>
// CHECK:     | |   `-ImplicitCastExpr {{.+}} 'void *__single(*__single)(va_list)' <LValueToRValue>
// CHECK:     | |     `-DeclRefExpr {{.+}} [[var_func_ptr]]
// CHECK:     | |-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK:     | | `-DeclRefExpr {{.+}} [[var_dst_str]]
// CHECK:     | |-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK:     | | `-DeclRefExpr {{.+}} [[var_src_str]]
// CHECK:     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |   `-DeclRefExpr {{.+}} [[var_len]]
// CHECK:     `-CallExpr
// CHECK:       |-ImplicitCastExpr {{.+}} 'void *(*__single)(void *(*)(va_list), ...)' <FunctionToPointerDecay>
// CHECK:       | `-DeclRefExpr {{.+}} [[func_static_1]]
// CHECK:       |-ImplicitCastExpr {{.+}} 'void *(*)(va_list)' <BoundsSafetyPointerCast>
// CHECK:       | `-ImplicitCastExpr {{.+}} 'void *(*__single)(char *)' <BitCast>
// CHECK:       |   `-ImplicitCastExpr {{.+}} 'void *__single(*__single)(va_list)' <LValueToRValue>
// CHECK:       |     `-DeclRefExpr {{.+}} [[var_func_ptr2]]
// CHECK:       |-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK:       | `-DeclRefExpr {{.+}} [[var_dst_str]]
// CHECK:       |-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK:       | `-DeclRefExpr {{.+}} [[var_src_str]]
// CHECK:       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:         `-DeclRefExpr {{.+}} [[var_len]]
