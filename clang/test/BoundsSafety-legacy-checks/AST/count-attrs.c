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
// CHECK:TranslationUnitDecl {{.*}}
// CHECK:|-FunctionDecl {{.*}} frob 'int *__single __counted_by(len)(int)'
// CHECK-NEXT:| `-ParmVarDecl {{.*}} used len 'int'
// CHECK-NEXT:|-FunctionDecl {{.*}} byte_frob 'int *__single __sized_by(len)(int)'
// CHECK-NEXT:| `-ParmVarDecl {{.*}} used len 'int'
// CHECK-NEXT:|-FunctionDecl {{.*}} alloc_bytes 'void *__single(int)'
// CHECK-NEXT:| |-ParmVarDecl {{.*}} byte_count 'int'
// CHECK-NEXT:| `-AllocSizeAttr {{.*}} 1
// CHECK-NEXT:|-TypedefDecl {{.*}} referenced int_array_t '__array_decay_discards_count_in_parameters int[10]':'int[10]'
// CHECK-NEXT:| `-MacroQualifiedType {{.*}} '__array_decay_discards_count_in_parameters int[10]' sugar
// CHECK-NEXT:|   `-AttributedType {{.*}} 'int[10] __attribute__((decay_discards_count_in_parameters))' sugar
// CHECK-NEXT:|     `-ParenType {{.*}} 'int[10]' sugar
// CHECK-NEXT:|       `-ConstantArrayType {{.*}} 'int[10]' 10
// CHECK-NEXT:|         `-BuiltinType {{.*}} 'int'
// CHECK-NEXT:|-FunctionDecl {{.*}} count_attr_in_bracket 'void (int *__single __counted_by(len), int)'
// CHECK-NEXT:| |-ParmVarDecl {{.*}} buf 'int *__single __counted_by(len)':'int *__single'
// CHECK-NEXT:| `-ParmVarDecl {{.*}} used len 'int'
// CHECK-NEXT:|   `-DependerDeclsAttr {{.*}} Implicit {{.*}} 0
// CHECK-NEXT:|-FunctionDecl {{.*}} count_ignored_from_array 'void (int *__single)'
// CHECK-NEXT:| `-ParmVarDecl {{.*}} buf 'int *__single'
// CHECK-NEXT:|-FunctionDecl {{.*}} count_ignored_and_attr 'void (int *__single __counted_by(count), int)'
// CHECK-NEXT:| |-ParmVarDecl {{.*}} buf 'int *__single __counted_by(count)':'int *__single'
// CHECK-NEXT:| `-ParmVarDecl {{.*}} used count 'int'
// CHECK-NEXT:|   `-DependerDeclsAttr {{.*}} Implicit {{.*}} 0
// CHECK-NEXT:|-RecordDecl {{.*}} struct s definition
// CHECK-NEXT:| |-FieldDecl {{.*}} bp 'int *__single __counted_by(l)':'int *__single'
// CHECK-NEXT:| |-FieldDecl {{.*}} bp2 'int *__single __counted_by(l + 1)':'int *__single'
// CHECK-NEXT:| |-FieldDecl {{.*}} bp3 'unsigned char *__single __sized_by(l)':'unsigned char *__single'
// CHECK-NEXT:| |-FieldDecl {{.*}} bp4 'unsigned char *__single __sized_by(l + 1)':'unsigned char *__single'
// CHECK-NEXT:| `-FieldDecl {{.*}} referenced l 'int'
// CHECK-NEXT:|   `-DependerDeclsAttr {{.*}} Implicit {{.*}} {{.*}} {{.*}} {{.*}} 0 0 0 0
// CHECK-NEXT:|-FunctionDecl {{.*}} test 'void (void)'
// CHECK-NEXT:| `-CompoundStmt {{.*}}
// CHECK-NEXT:|   |-DeclStmt {{.*}}
// CHECK-NEXT:|   | `-VarDecl {{.*}} used n 'int' cinit
// CHECK-NEXT:|   |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT:|   |   `-DependerDeclsAttr {{.*}} Implicit {{.*}} 0
// CHECK-NEXT:|   |-DeclStmt {{.*}}
// CHECK-NEXT:|   | `-VarDecl {{.*}} buf1 'int *__single __counted_by(n)':'int *__single'
// CHECK-NEXT:|   |-DeclStmt {{.*}}
// CHECK-NEXT:|   | `-VarDecl {{.*}} used n2 'int' cinit
// CHECK-NEXT:|   |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT:|   |   `-DependerDeclsAttr {{.*}} Implicit {{.*}} 0
// CHECK-NEXT:|   |-DeclStmt {{.*}}
// CHECK-NEXT:|   | `-VarDecl {{.*}} buf2 'int *__single __counted_by(n2)':'int *__single'
// CHECK-NEXT:|   |-DeclStmt {{.*}}
// CHECK-NEXT:|   | `-VarDecl {{.*}} used n3 'int' cinit
// CHECK-NEXT:|   |   |-ImplicitCastExpr {{.*}} 'int' <IntegralCast>
// CHECK-NEXT:|   |   | `-BinaryOperator {{.*}} 'unsigned long' '*'
// CHECK-NEXT:|   |   |   |-UnaryExprOrTypeTraitExpr {{.*}} 'unsigned long' sizeof 'int'
// CHECK-NEXT:|   |   |   `-ImplicitCastExpr {{.*}} 'unsigned long' <IntegralCast>
// CHECK-NEXT:|   |   |     `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:|   |   |       `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'n' 'int'
// CHECK-NEXT:|   |   `-DependerDeclsAttr {{.*}} Implicit {{.*}} 0
// CHECK-NEXT:|   |-DeclStmt {{.*}}
// CHECK-NEXT:|   | `-VarDecl {{.*}} byte_buf1 'unsigned char *__single __sized_by(n3)':'unsigned char *__single'
// CHECK-NEXT:|   |-DeclStmt {{.*}}
// CHECK-NEXT:|   | `-VarDecl {{.*}} used n4 'int' cinit
// CHECK-NEXT:|   |   |-ImplicitCastExpr {{.*}} 'int' <IntegralCast>
// CHECK-NEXT:|   |   | `-BinaryOperator {{.*}} 'unsigned long' '*'
// CHECK-NEXT:|   |   |   |-UnaryExprOrTypeTraitExpr {{.*}} 'unsigned long' sizeof 'int'
// CHECK-NEXT:|   |   |   `-ImplicitCastExpr {{.*}} 'unsigned long' <IntegralCast>
// CHECK-NEXT:|   |   |     `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:|   |   |       `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'n2' 'int'
// CHECK-NEXT:|   |   `-DependerDeclsAttr {{.*}} Implicit {{.*}} 0
// CHECK-NEXT:|   `-DeclStmt {{.*}}
// CHECK-NEXT:|     `-VarDecl {{.*}} byte_buf2 'unsigned char *__single __sized_by(n4)':'unsigned char *__single'
// CHECK-NEXT:|-FunctionDecl {{.*}} frob_body 'int *__single __counted_by(len)(int)'
// CHECK-NEXT:| |-ParmVarDecl {{.*}} used len 'int'
// CHECK-NEXT:| `-CompoundStmt {{.*}}
// CHECK-NEXT:|   `-ReturnStmt {{.*}}
// CHECK-NEXT:|     `-ImplicitCastExpr {{.*}} 'int *__single __counted_by(len)':'int *__single' <NullToPointer>
// CHECK-NEXT:|       `-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT:`-FunctionDecl {{.*}} byte_frob_body 'int *__single __sized_by(len)(int)'
// CHECK-NEXT:  |-ParmVarDecl {{.*}} used len 'int'
// CHECK-NEXT:  `-CompoundStmt {{.*}}
// CHECK-NEXT:    `-ReturnStmt {{.*}}
// CHECK-NEXT:      `-ImplicitCastExpr {{.*}} 'int *__single __sized_by(len)':'int *__single' <NullToPointer>
// CHECK-NEXT:        `-IntegerLiteral {{.*}} 'int' 0
