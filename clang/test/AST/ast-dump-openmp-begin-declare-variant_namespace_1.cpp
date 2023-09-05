// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++ | FileCheck %s

namespace A {
int foo(void) { // expected-note {{candidate function}}
  return 0;
}
} // namespace A

namespace B {
int bar(void) {
  return 1;
}
} // namespace B

namespace C {
int baz(void) {
  return 2;
}
} // namespace C

#pragma omp begin declare variant match(implementation = {vendor(llvm)})

// This will *not* be a specialization of A::foo(void).
int foo(void) { // expected-note {{candidate function}}
  return 3;
}

namespace B {
// This will *not* be a specialization of A::foo(void).
int foo(void) {
  return 4;
}
// This will be a specialization of B::bar(void).
int bar(void) {
  return 0;
}
} // namespace B

using namespace C;

// This will be a specialization of C::baz(void).
int baz(void) {
  return 0;
}
#pragma omp end declare variant


int explicit1() {
  // Should return 0.
  return A::foo() + B::bar() + C::baz();
}

int implicit2() {
  using namespace A;
  using namespace B;
  // Should return 0.
  foo(); // expected-error {{call to 'foo' is ambiguous}}
  return bar() + baz();
}

int main() {
  // Should return 0.
  return explicit1() + implicit2();
}

// CHECK:      |-NamespaceDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:3:11 A
// CHECK-NEXT: | `-FunctionDecl [[ADDR_1:0x[a-z0-9]*]] <line:4:1, line:6:1> line:4:5 used foo 'int ({{.*}})'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_2:0x[a-z0-9]*]] <col:15, line:6:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_3:0x[a-z0-9]*]] <line:5:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_4:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-NamespaceDecl [[ADDR_5:0x[a-z0-9]*]] <line:9:1, line:13:1> line:9:11 B
// CHECK-NEXT: | `-FunctionDecl [[ADDR_6:0x[a-z0-9]*]] <line:10:1, line:12:1> line:10:5 used bar 'int ({{.*}})'
// CHECK-NEXT: |   |-CompoundStmt [[ADDR_7:0x[a-z0-9]*]] <col:15, line:12:1>
// CHECK-NEXT: |   | `-ReturnStmt [[ADDR_8:0x[a-z0-9]*]] <line:11:3, col:10>
// CHECK-NEXT: |   |   `-IntegerLiteral [[ADDR_9:0x[a-z0-9]*]] <col:10> 'int' 1
// CHECK-NEXT: |   `-OMPDeclareVariantAttr [[ADDR_10:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CHECK-NEXT: |     `-DeclRefExpr [[ADDR_11:0x[a-z0-9]*]] <line:34:1> 'int ({{.*}})' Function [[ADDR_12:0x[a-z0-9]*]] 'bar[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: |-NamespaceDecl [[ADDR_13:0x[a-z0-9]*]] <line:15:1, line:19:1> line:15:11 C
// CHECK-NEXT: | `-FunctionDecl [[ADDR_14:0x[a-z0-9]*]] <line:16:1, line:18:1> line:16:5 used baz 'int ({{.*}})'
// CHECK-NEXT: |   |-CompoundStmt [[ADDR_15:0x[a-z0-9]*]] <col:15, line:18:1>
// CHECK-NEXT: |   | `-ReturnStmt [[ADDR_16:0x[a-z0-9]*]] <line:17:3, col:10>
// CHECK-NEXT: |   |   `-IntegerLiteral [[ADDR_17:0x[a-z0-9]*]] <col:10> 'int' 2
// CHECK-NEXT: |   `-OMPDeclareVariantAttr [[ADDR_18:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CHECK-NEXT: |     `-DeclRefExpr [[ADDR_19:0x[a-z0-9]*]] <line:42:1> 'int ({{.*}})' Function [[ADDR_20:0x[a-z0-9]*]] 'baz[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_21:0x[a-z0-9]*]] <line:24:1, col:13> col:5 implicit foo 'int ({{.*}})'
// CHECK-NEXT: | |-OMPDeclareVariantAttr [[ADDR_22:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CHECK-NEXT: | | `-DeclRefExpr [[ADDR_23:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_24:0x[a-z0-9]*]] 'foo[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: | `-OMPDeclareVariantAttr [[ADDR_25:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CHECK-NEXT: |   `-DeclRefExpr [[ADDR_26:0x[a-z0-9]*]] <line:30:1> 'int ({{.*}})' Function [[ADDR_27:0x[a-z0-9]*]] 'foo[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_24]] <line:24:1, line:26:1> line:24:1 foo[implementation={vendor(llvm)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_28:0x[a-z0-9]*]] <col:15, line:26:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_29:0x[a-z0-9]*]] <line:25:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_30:0x[a-z0-9]*]] <col:10> 'int' 3
// CHECK-NEXT: |-NamespaceDecl [[ADDR_31:0x[a-z0-9]*]] prev [[ADDR_5]] <line:28:1, line:37:1> line:28:11 B
// CHECK-NEXT: | |-original Namespace [[ADDR_5]] 'B'
// CHECK-NEXT: | |-FunctionDecl [[ADDR_27]] <line:30:1, line:32:1> line:30:1 foo[implementation={vendor(llvm)}] 'int ({{.*}})'
// CHECK-NEXT: | | `-CompoundStmt [[ADDR_32:0x[a-z0-9]*]] <col:15, line:32:1>
// CHECK-NEXT: | |   `-ReturnStmt [[ADDR_33:0x[a-z0-9]*]] <line:31:3, col:10>
// CHECK-NEXT: | |     `-IntegerLiteral [[ADDR_34:0x[a-z0-9]*]] <col:10> 'int' 4
// CHECK-NEXT: | `-FunctionDecl [[ADDR_12]] <line:34:1, line:36:1> line:34:1 bar[implementation={vendor(llvm)}] 'int ({{.*}})'
// CHECK-NEXT: |   `-CompoundStmt [[ADDR_35:0x[a-z0-9]*]] <col:15, line:36:1>
// CHECK-NEXT: |     `-ReturnStmt [[ADDR_36:0x[a-z0-9]*]] <line:35:3, col:10>
// CHECK-NEXT: |       `-IntegerLiteral [[ADDR_37:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-UsingDirectiveDecl [[ADDR_38:0x[a-z0-9]*]] <line:39:1, col:17> col:17 Namespace [[ADDR_13]] 'C'
// CHECK-NEXT: |-FunctionDecl [[ADDR_20]] <line:42:1, line:44:1> line:42:1 baz[implementation={vendor(llvm)}] 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_39:0x[a-z0-9]*]] <col:15, line:44:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_40:0x[a-z0-9]*]] <line:43:3, col:10>
// CHECK-NEXT: |     `-IntegerLiteral [[ADDR_41:0x[a-z0-9]*]] <col:10> 'int' 0
// CHECK-NEXT: |-FunctionDecl [[ADDR_42:0x[a-z0-9]*]] <line:48:1, line:51:1> line:48:5 used explicit1 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_43:0x[a-z0-9]*]] <col:17, line:51:1>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_44:0x[a-z0-9]*]] <line:50:3, col:39>
// CHECK-NEXT: |     `-BinaryOperator [[ADDR_45:0x[a-z0-9]*]] <col:10, col:39> 'int' '+'
// CHECK-NEXT: |       |-BinaryOperator [[ADDR_46:0x[a-z0-9]*]] <col:10, col:28> 'int' '+'
// CHECK-NEXT: |       | |-CallExpr [[ADDR_47:0x[a-z0-9]*]] <col:10, col:17> 'int'
// CHECK-NEXT: |       | | `-ImplicitCastExpr [[ADDR_48:0x[a-z0-9]*]] <col:10, col:13> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT: |       | |   `-DeclRefExpr [[ADDR_49:0x[a-z0-9]*]] <col:10, col:13> 'int ({{.*}})' {{.*}}Function [[ADDR_1]] 'foo' 'int ({{.*}})'
// CHECK-NEXT: |       | |   `-NestedNameSpecifier Namespace [[ADDR_0]] 'A'
// CHECK-NEXT: |       | `-PseudoObjectExpr [[ADDR_50:0x[a-z0-9]*]] <col:21, col:28> 'int'
// CHECK-NEXT: |       |   |-CallExpr [[ADDR_51:0x[a-z0-9]*]] <col:21, col:28> 'int'
// CHECK-NEXT: |       |   | `-ImplicitCastExpr [[ADDR_52:0x[a-z0-9]*]] <col:21, col:24> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT: |       |   |   `-DeclRefExpr [[ADDR_53:0x[a-z0-9]*]] <col:21, col:24> 'int ({{.*}})' {{.*}}Function [[ADDR_6]] 'bar' 'int ({{.*}})'
// CHECK-NEXT: |       |   |   `-NestedNameSpecifier Namespace [[ADDR_31]] 'B'
// CHECK-NEXT: |       |   `-CallExpr [[ADDR_54:0x[a-z0-9]*]] <line:34:1, line:50:28> 'int'
// CHECK-NEXT: |       |     `-ImplicitCastExpr [[ADDR_55:0x[a-z0-9]*]] <line:34:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT: |       |       `-DeclRefExpr [[ADDR_11]] <col:1> 'int ({{.*}})' Function [[ADDR_12]] 'bar[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: |       `-PseudoObjectExpr [[ADDR_56:0x[a-z0-9]*]] <line:50:32, col:39> 'int'
// CHECK-NEXT: |         |-CallExpr [[ADDR_57:0x[a-z0-9]*]] <col:32, col:39> 'int'
// CHECK-NEXT: |         | `-ImplicitCastExpr [[ADDR_58:0x[a-z0-9]*]] <col:32, col:35> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT: |         |   `-DeclRefExpr [[ADDR_59:0x[a-z0-9]*]] <col:32, col:35> 'int ({{.*}})' {{.*}}Function [[ADDR_14]] 'baz' 'int ({{.*}})'
// CHECK-NEXT: |         |   `-NestedNameSpecifier Namespace [[ADDR_13]] 'C'
// CHECK-NEXT: |         `-CallExpr [[ADDR_60:0x[a-z0-9]*]] <line:42:1, line:50:39> 'int'
// CHECK-NEXT: |           `-ImplicitCastExpr [[ADDR_61:0x[a-z0-9]*]] <line:42:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT: |             `-DeclRefExpr [[ADDR_19]] <col:1> 'int ({{.*}})' Function [[ADDR_20]] 'baz[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_62:0x[a-z0-9]*]] <line:53:1, line:59:1> line:53:5 used implicit2 'int ({{.*}})'
// CHECK-NEXT: | `-CompoundStmt [[ADDR_63:0x[a-z0-9]*]] <col:17, line:59:1>
// CHECK-NEXT: |   |-DeclStmt [[ADDR_64:0x[a-z0-9]*]] <line:54:3, col:20>
// CHECK-NEXT: |   | `-UsingDirectiveDecl [[ADDR_65:0x[a-z0-9]*]] <col:3, col:19> col:19 Namespace [[ADDR_0]] 'A'
// CHECK-NEXT: |   |-DeclStmt [[ADDR_66:0x[a-z0-9]*]] <line:55:3, col:20>
// CHECK-NEXT: |   | `-UsingDirectiveDecl [[ADDR_67:0x[a-z0-9]*]] <col:3, col:19> col:19 Namespace [[ADDR_5]] 'B'
// CHECK-NEXT: |   |-RecoveryExpr {{.*}} <line:57:3, col:7>
// CHECK-NEXT: |   | `-UnresolvedLookupExpr {{.*}} <col:3>
// CHECK-NEXT: |   `-ReturnStmt [[ADDR_68:0x[a-z0-9]*]] <line:58:3, col:22>
// CHECK-NEXT: |     `-BinaryOperator [[ADDR_69:0x[a-z0-9]*]] <col:10, col:22> 'int' '+'
// CHECK-NEXT: |       |-PseudoObjectExpr [[ADDR_70:0x[a-z0-9]*]] <col:10, col:14> 'int'
// CHECK-NEXT: |       | |-CallExpr [[ADDR_71:0x[a-z0-9]*]] <col:10, col:14> 'int'
// CHECK-NEXT: |       | | `-ImplicitCastExpr [[ADDR_72:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT: |       | |   `-DeclRefExpr [[ADDR_73:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_6]] 'bar' 'int ({{.*}})'
// CHECK-NEXT: |       | `-CallExpr [[ADDR_74:0x[a-z0-9]*]] <line:34:1, line:58:14> 'int'
// CHECK-NEXT: |       |   `-ImplicitCastExpr [[ADDR_75:0x[a-z0-9]*]] <line:34:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT: |       |     `-DeclRefExpr [[ADDR_11]] <col:1> 'int ({{.*}})' Function [[ADDR_12]] 'bar[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: |       `-PseudoObjectExpr [[ADDR_76:0x[a-z0-9]*]] <line:58:18, col:22> 'int'
// CHECK-NEXT: |         |-CallExpr [[ADDR_77:0x[a-z0-9]*]] <col:18, col:22> 'int'
// CHECK-NEXT: |         | `-ImplicitCastExpr [[ADDR_78:0x[a-z0-9]*]] <col:18> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT: |         |   `-DeclRefExpr [[ADDR_79:0x[a-z0-9]*]] <col:18> 'int ({{.*}})' {{.*}}Function [[ADDR_14]] 'baz' 'int ({{.*}})'
// CHECK-NEXT: |         `-CallExpr [[ADDR_80:0x[a-z0-9]*]] <line:42:1, line:58:22> 'int'
// CHECK-NEXT: |           `-ImplicitCastExpr [[ADDR_81:0x[a-z0-9]*]] <line:42:1> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT: |             `-DeclRefExpr [[ADDR_19]] <col:1> 'int ({{.*}})' Function [[ADDR_20]] 'baz[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CHECK-NEXT: `-FunctionDecl [[ADDR_82:0x[a-z0-9]*]] <line:61:1, line:64:1> line:61:5 main 'int ({{.*}})'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_83:0x[a-z0-9]*]] <col:12, line:64:1>
// CHECK-NEXT:     `-ReturnStmt [[ADDR_84:0x[a-z0-9]*]] <line:63:3, col:34>
// CHECK-NEXT:       `-BinaryOperator [[ADDR_85:0x[a-z0-9]*]] <col:10, col:34> 'int' '+'
// CHECK-NEXT:         |-CallExpr [[ADDR_86:0x[a-z0-9]*]] <col:10, col:20> 'int'
// CHECK-NEXT:         | `-ImplicitCastExpr [[ADDR_87:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         |   `-DeclRefExpr [[ADDR_88:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_42]] 'explicit1' 'int ({{.*}})'
// CHECK-NEXT:         `-CallExpr [[ADDR_89:0x[a-z0-9]*]] <col:24, col:34> 'int'
// CHECK-NEXT:           `-ImplicitCastExpr [[ADDR_90:0x[a-z0-9]*]] <col:24> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:             `-DeclRefExpr [[ADDR_91:0x[a-z0-9]*]] <col:24> 'int ({{.*}})' {{.*}}Function [[ADDR_62]] 'implicit2' 'int ({{.*}})'