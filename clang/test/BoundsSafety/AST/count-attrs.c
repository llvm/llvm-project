// RUN: %clang_cc1 -triple x86_64-apple-mac -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-mac -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s
#include <ptrcheck.h>

int *__counted_by(len) frob(int len);

int *__sized_by(len) byte_frob(int len);
void *alloc_bytes(int byte_count) __attribute__((alloc_size(1)));
// <rdar://problem/70626464>
// void *alloc_items(int byte_count, int size) __attribute__((alloc_size(1, 2)));

typedef int (__array_decay_discards_count_in_parameters int_array_t)[10];
void count_attr_in_bracket(int buf[__counted_by(len)], int len);
void count_ignored_from_array(int (__array_decay_discards_count_in_parameters buf)[10]);
void count_ignored_and_attr(int_array_t __counted_by(count) buf, int count);

struct s {
  int *__counted_by(l) bp;
  int *bp2 __counted_by(l+1);
  unsigned char *bp3 __sized_by(l);
  unsigned char *bp4 __sized_by(l+1);
  int l;
};

void test(void) {
  int n = 0;
  int *__counted_by(n) buf1;
  int n2 = 0;
  int *buf2 __counted_by(n2);
  int n3 = sizeof(int) * n;
  unsigned char *__sized_by(n3) byte_buf1;
  int n4 = sizeof(int) * n2;
  unsigned char *byte_buf2 __sized_by(n4);
}

int *__counted_by(len) frob_body(int len) { return 0; }
int *__sized_by(len) byte_frob_body(int len) { return 0; }
// CHECK:      {{^}}TranslationUnitDecl
// CHECK:      {{^}}|-FunctionDecl [[func_frob:0x[^ ]+]] {{.+}} frob
// CHECK-NEXT: {{^}}| `-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK-NEXT: {{^}}|-FunctionDecl [[func_byte_frob:0x[^ ]+]] {{.+}} byte_frob
// CHECK-NEXT: {{^}}| `-ParmVarDecl [[var_len_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|-FunctionDecl [[func_alloc_bytes:0x[^ ]+]] {{.+}} alloc_bytes
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_byte_count:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-AllocSizeAttr
// CHECK-NEXT: {{^}}|-TypedefDecl
// CHECK-NEXT: {{^}}| `-MacroQualifiedType
// CHECK-NEXT: {{^}}|   `-AttributedType
// CHECK-NEXT: {{^}}|     `-ParenType
// CHECK-NEXT: {{^}}|       `-ConstantArrayType
// CHECK: {{^}}|-FunctionDecl [[func_count_attr_in_bracket:0x[^ ]+]] {{.+}} count_attr_in_bracket
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_buf:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-ParmVarDecl [[var_len_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|-FunctionDecl [[func_count_ignored_from_array:0x[^ ]+]] {{.+}} count_ignored_from_array
// CHECK-NEXT: {{^}}| `-ParmVarDecl [[var_buf_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|-FunctionDecl [[func_count_ignored_and_attr:0x[^ ]+]] {{.+}} count_ignored_and_attr
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_buf_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-ParmVarDecl [[var_count:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|-RecordDecl
// CHECK-NEXT: {{^}}| |-FieldDecl
// CHECK-NEXT: {{^}}| |-FieldDecl
// CHECK-NEXT: {{^}}| |-FieldDecl
// CHECK-NEXT: {{^}}| |-FieldDecl
// CHECK-NEXT: {{^}}| `-FieldDecl
// CHECK-NEXT: {{^}}|   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|-FunctionDecl [[func_test:0x[^ ]+]] {{.+}} test
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_n:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_buf1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_n2:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_buf2:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_n3:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   |-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK-NEXT: {{^}}|   |   | `-BinaryOperator {{.+}} 'unsigned long' '*'
// CHECK-NEXT: {{^}}|   |   |   |-UnaryExprOrTypeTraitExpr
// CHECK-NEXT: {{^}}|   |   |   `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK-NEXT: {{^}}|   |   |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |   |       `-DeclRefExpr {{.+}} [[var_n]]
// CHECK-NEXT: {{^}}|   |   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_byte_buf1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_n4:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   |-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK-NEXT: {{^}}|   |   | `-BinaryOperator {{.+}} 'unsigned long' '*'
// CHECK-NEXT: {{^}}|   |   |   |-UnaryExprOrTypeTraitExpr
// CHECK-NEXT: {{^}}|   |   |   `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK-NEXT: {{^}}|   |   |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |   |       `-DeclRefExpr {{.+}} [[var_n2]]
// CHECK-NEXT: {{^}}|   |   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|   `-DeclStmt
// CHECK-NEXT: {{^}}|     `-VarDecl [[var_byte_buf2:0x[^ ]+]]
// CHECK-NEXT: {{^}}|-FunctionDecl [[func_frob_body:0x[^ ]+]] {{.+}} frob_body
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'len == 0'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|       |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|       | |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       |-OpaqueValueExpr [[ove]]
// CHECK-NEXT: {{^}}|       | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|       `-OpaqueValueExpr [[ove_1]]
// CHECK-NEXT: {{^}}|         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           `-DeclRefExpr {{.+}} [[var_len_3]]
// CHECK-NEXT: {{^}}`-FunctionDecl [[func_byte_frob_body:0x[^ ]+]] {{.+}} byte_frob_body
// CHECK-NEXT: {{^}}  |-ParmVarDecl [[var_len_4:0x[^ ]+]]
// CHECK-NEXT: {{^}}  `-CompoundStmt
// CHECK-NEXT: {{^}}    `-ReturnStmt
// CHECK-NEXT: {{^}}      `-BoundsCheckExpr {{.+}} 'len == 0'
// CHECK-NEXT: {{^}}        |-ImplicitCastExpr {{.+}} 'int *__single __sized_by(len)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}        | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}        |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}        | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}        | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}        |-OpaqueValueExpr [[ove_2]]
// CHECK-NEXT: {{^}}        | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}        `-OpaqueValueExpr [[ove_3]]
// CHECK-NEXT: {{^}}          `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}            `-DeclRefExpr {{.+}} [[var_len_4]]
