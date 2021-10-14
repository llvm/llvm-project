// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s       | FileCheck %s --check-prefix=C
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -ast-dump %s -x c++| FileCheck %s --check-prefix=CXX
// expected-no-diagnostics

int also_before(void) {
  return 0;
}

#pragma omp begin declare variant match(implementation={vendor(llvm)})
int also_after(void) {
  return 1;
}
int also_before(void) {
  return 2;
}
#pragma omp end declare variant

int also_after(void) {
  return 0;
}

void foo();
typedef int(*fd)(void);
int main() {
  // Should return 0.
  fd fns[2];
  fns[0] = &also_before;
  fns[1] = also_after;
  return (foo(), also_after)() +
         (fns[0])() +
         (1[fns])();
}

// Make sure:
//  - we see the specialization in the AST
//  - we pick the right callees

// C:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:5:5 used also_before 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:23, line:7:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:6:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_4:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_5:0x[a-z0-9]*]] <line:13:1> 'int ({{.*}})' Function [[ADDR_6:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:10:1, col:20> col:5 implicit used also_after 'int ({{.*}})'
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_8:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_9:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_10:0x[a-z0-9]*]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_10]] <col:1, line:12:1> line:10:1 also_after[implementation={vendor(llvm)}] 'int ({{.*}})'
// C-NEXT: | `-CompoundStmt [[ADDR_11:0x[a-z0-9]*]] <col:22, line:12:1>
// C-NEXT: |   `-ReturnStmt [[ADDR_12:0x[a-z0-9]*]] <line:11:3, col:10>
// C-NEXT: |     `-IntegerLiteral [[ADDR_13:0x[a-z0-9]*]] <col:10> 'int' 1
// C-NEXT: |-FunctionDecl [[ADDR_6]] <line:13:1, line:15:1> line:13:1 also_before[implementation={vendor(llvm)}] 'int ({{.*}})'
// C-NEXT: | `-CompoundStmt [[ADDR_14:0x[a-z0-9]*]] <col:23, line:15:1>
// C-NEXT: |   `-ReturnStmt [[ADDR_15:0x[a-z0-9]*]] <line:14:3, col:10>
// C-NEXT: |     `-IntegerLiteral [[ADDR_16:0x[a-z0-9]*]] <col:10> 'int' 2
// C-NEXT: |-FunctionDecl [[ADDR_17:0x[a-z0-9]*]] prev [[ADDR_7]] <line:18:1, line:20:1> line:18:5 used also_after 'int ({{.*}})'
// C-NEXT: | |-CompoundStmt [[ADDR_18:0x[a-z0-9]*]] <col:22, line:20:1>
// C-NEXT: | | `-ReturnStmt [[ADDR_19:0x[a-z0-9]*]] <line:19:3, col:10>
// C-NEXT: | |   `-IntegerLiteral [[ADDR_20:0x[a-z0-9]*]] <col:10> 'int' 0
// C-NEXT: | `-OMPDeclareVariantAttr [[ADDR_21:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(llvm)}
// C-NEXT: |   `-DeclRefExpr [[ADDR_9]] <line:10:1> 'int ({{.*}})' Function [[ADDR_10]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// C-NEXT: |-FunctionDecl [[ADDR_22:0x[a-z0-9]*]] <line:22:1, col:10> col:6 used foo 'void ({{.*}})'
// C-NEXT: |-TypedefDecl [[ADDR_23:0x[a-z0-9]*]] <line:23:1, col:22> col:14 referenced fd 'int (*)({{.*}})'
// C-NEXT: | `-PointerType [[ADDR_24:0x[a-z0-9]*]] 'int (*)({{.*}})'
// C-NEXT: |   `-ParenType [[ADDR_25:0x[a-z0-9]*]] 'int ({{.*}})' sugar
// C-NEXT: |     `-FunctionProtoType [[ADDR_26:0x[a-z0-9]*]] 'int ({{.*}})' cdecl
// C-NEXT: |       `-BuiltinType [[ADDR_27:0x[a-z0-9]*]] 'int'
// C-NEXT: `-FunctionDecl [[ADDR_28:0x[a-z0-9]*]] <line:24:1, line:32:1> line:24:5 main 'int ({{.*}})'
// C-NEXT:   `-CompoundStmt [[ADDR_29:0x[a-z0-9]*]] <col:12, line:32:1>
// C-NEXT:     |-DeclStmt [[ADDR_30:0x[a-z0-9]*]] <line:26:3, col:12>
// C-NEXT:     | `-VarDecl [[ADDR_31:0x[a-z0-9]*]] <col:3, col:11> col:6 used fns 'fd[2]'
// C-NEXT:     |-BinaryOperator [[ADDR_32:0x[a-z0-9]*]] <line:27:3, col:13> 'fd':'int (*)({{.*}})' '='
// C-NEXT:     | |-ArraySubscriptExpr [[ADDR_33:0x[a-z0-9]*]] <col:3, col:8> 'fd':'int (*)({{.*}})' lvalue
// C-NEXT:     | | |-ImplicitCastExpr [[ADDR_34:0x[a-z0-9]*]] <col:3> 'fd *' <ArrayToPointerDecay>
// C-NEXT:     | | | `-DeclRefExpr [[ADDR_35:0x[a-z0-9]*]] <col:3> 'fd[2]' {{.*}}Var [[ADDR_31]] 'fns' 'fd[2]'
// C-NEXT:     | | `-IntegerLiteral [[ADDR_36:0x[a-z0-9]*]] <col:7> 'int' 0
// C-NEXT:     | `-UnaryOperator [[ADDR_37:0x[a-z0-9]*]] <col:12, col:13> 'int (*)({{.*}})' prefix '&' cannot overflow
// C-NEXT:     |   `-DeclRefExpr [[ADDR_38:0x[a-z0-9]*]] <col:13> 'int ({{.*}})' Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
// C-NEXT:     |-BinaryOperator [[ADDR_39:0x[a-z0-9]*]] <line:28:3, col:12> 'fd':'int (*)({{.*}})' '='
// C-NEXT:     | |-ArraySubscriptExpr [[ADDR_40:0x[a-z0-9]*]] <col:3, col:8> 'fd':'int (*)({{.*}})' lvalue
// C-NEXT:     | | |-ImplicitCastExpr [[ADDR_41:0x[a-z0-9]*]] <col:3> 'fd *' <ArrayToPointerDecay>
// C-NEXT:     | | | `-DeclRefExpr [[ADDR_42:0x[a-z0-9]*]] <col:3> 'fd[2]' {{.*}}Var [[ADDR_31]] 'fns' 'fd[2]'
// C-NEXT:     | | `-IntegerLiteral [[ADDR_43:0x[a-z0-9]*]] <col:7> 'int' 1
// C-NEXT:     | `-ImplicitCastExpr [[ADDR_44:0x[a-z0-9]*]] <col:12> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:     |   `-DeclRefExpr [[ADDR_45:0x[a-z0-9]*]] <col:12> 'int ({{.*}})' Function [[ADDR_17]] 'also_after' 'int ({{.*}})'
// C-NEXT:     `-ReturnStmt [[ADDR_46:0x[a-z0-9]*]] <line:29:3, line:31:19>
// C-NEXT:       `-BinaryOperator [[ADDR_47:0x[a-z0-9]*]] <line:29:10, line:31:19> 'int' '+'
// C-NEXT:         |-BinaryOperator [[ADDR_48:0x[a-z0-9]*]] <line:29:10, line:30:19> 'int' '+'
// C-NEXT:         | |-CallExpr [[ADDR_49:0x[a-z0-9]*]] <line:29:10, col:30> 'int'
// C-NEXT:         | | `-ParenExpr [[ADDR_50:0x[a-z0-9]*]] <col:10, col:28> 'int (*)({{.*}})'
// C-NEXT:         | |   `-BinaryOperator [[ADDR_51:0x[a-z0-9]*]] <col:11, col:18> 'int (*)({{.*}})' ','
// C-NEXT:         | |     |-CallExpr [[ADDR_52:0x[a-z0-9]*]] <col:11, col:15> 'void'
// C-NEXT:         | |     | `-ImplicitCastExpr [[ADDR_53:0x[a-z0-9]*]] <col:11> 'void (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | |     |   `-DeclRefExpr [[ADDR_54:0x[a-z0-9]*]] <col:11> 'void ({{.*}})' Function [[ADDR_22]] 'foo' 'void ({{.*}})'
// C-NEXT:         | |     `-ImplicitCastExpr [[ADDR_55:0x[a-z0-9]*]] <col:18> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:         | |       `-DeclRefExpr [[ADDR_56:0x[a-z0-9]*]] <col:18> 'int ({{.*}})' Function [[ADDR_17]] 'also_after' 'int ({{.*}})'
// C-NEXT:         | `-CallExpr [[ADDR_57:0x[a-z0-9]*]] <line:30:10, col:19> 'int'
// C-NEXT:         |   `-ImplicitCastExpr [[ADDR_58:0x[a-z0-9]*]] <col:10, col:17> 'fd':'int (*)({{.*}})' <LValueToRValue>
// C-NEXT:         |     `-ParenExpr [[ADDR_59:0x[a-z0-9]*]] <col:10, col:17> 'fd':'int (*)({{.*}})' lvalue
// C-NEXT:         |       `-ArraySubscriptExpr [[ADDR_60:0x[a-z0-9]*]] <col:11, col:16> 'fd':'int (*)({{.*}})' lvalue
// C-NEXT:         |         |-ImplicitCastExpr [[ADDR_61:0x[a-z0-9]*]] <col:11> 'fd *' <ArrayToPointerDecay>
// C-NEXT:         |         | `-DeclRefExpr [[ADDR_62:0x[a-z0-9]*]] <col:11> 'fd[2]' {{.*}}Var [[ADDR_31]] 'fns' 'fd[2]'
// C-NEXT:         |         `-IntegerLiteral [[ADDR_63:0x[a-z0-9]*]] <col:15> 'int' 0
// C-NEXT:         `-CallExpr [[ADDR_64:0x[a-z0-9]*]] <line:31:10, col:19> 'int'
// C-NEXT:           `-ImplicitCastExpr [[ADDR_65:0x[a-z0-9]*]] <col:10, col:17> 'fd':'int (*)({{.*}})' <LValueToRValue>
// C-NEXT:             `-ParenExpr [[ADDR_66:0x[a-z0-9]*]] <col:10, col:17> 'fd':'int (*)({{.*}})' lvalue
// C-NEXT:               `-ArraySubscriptExpr [[ADDR_67:0x[a-z0-9]*]] <col:11, col:16> 'fd':'int (*)({{.*}})' lvalue
// C-NEXT:                 |-IntegerLiteral [[ADDR_68:0x[a-z0-9]*]] <col:11> 'int' 1
// C-NEXT:                 `-ImplicitCastExpr [[ADDR_69:0x[a-z0-9]*]] <col:13> 'fd *' <ArrayToPointerDecay>
// C-NEXT:                   `-DeclRefExpr [[ADDR_70:0x[a-z0-9]*]] <col:13> 'fd[2]' {{.*}}Var [[ADDR_31]] 'fns' 'fd[2]'

// CXX:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, line:7:1> line:5:5 used also_before 'int ({{.*}})'
// CXX-NEXT: | |-CompoundStmt [[ADDR_1:0x[a-z0-9]*]] <col:23, line:7:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_2:0x[a-z0-9]*]] <line:6:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_3:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_4:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_5:0x[a-z0-9]*]] <line:13:1> 'int ({{.*}})' Function [[ADDR_6:0x[a-z0-9]*]] 'also_before[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT: |-FunctionDecl [[ADDR_7:0x[a-z0-9]*]] <line:10:1, col:20> col:5 implicit used also_after 'int ({{.*}})'
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_8:0x[a-z0-9]*]] <<invalid sloc>> Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_9:0x[a-z0-9]*]] <col:1> 'int ({{.*}})' Function [[ADDR_10:0x[a-z0-9]*]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT: |-FunctionDecl [[ADDR_10]] <col:1, line:12:1> line:10:1 also_after[implementation={vendor(llvm)}] 'int ({{.*}})'
// CXX-NEXT: | `-CompoundStmt [[ADDR_11:0x[a-z0-9]*]] <col:22, line:12:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_12:0x[a-z0-9]*]] <line:11:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_13:0x[a-z0-9]*]] <col:10> 'int' 1
// CXX-NEXT: |-FunctionDecl [[ADDR_6]] <line:13:1, line:15:1> line:13:1 also_before[implementation={vendor(llvm)}] 'int ({{.*}})'
// CXX-NEXT: | `-CompoundStmt [[ADDR_14:0x[a-z0-9]*]] <col:23, line:15:1>
// CXX-NEXT: |   `-ReturnStmt [[ADDR_15:0x[a-z0-9]*]] <line:14:3, col:10>
// CXX-NEXT: |     `-IntegerLiteral [[ADDR_16:0x[a-z0-9]*]] <col:10> 'int' 2
// CXX-NEXT: |-FunctionDecl [[ADDR_17:0x[a-z0-9]*]] prev [[ADDR_7]] <line:18:1, line:20:1> line:18:5 used also_after 'int ({{.*}})'
// CXX-NEXT: | |-CompoundStmt [[ADDR_18:0x[a-z0-9]*]] <col:22, line:20:1>
// CXX-NEXT: | | `-ReturnStmt [[ADDR_19:0x[a-z0-9]*]] <line:19:3, col:10>
// CXX-NEXT: | |   `-IntegerLiteral [[ADDR_20:0x[a-z0-9]*]] <col:10> 'int' 0
// CXX-NEXT: | `-OMPDeclareVariantAttr [[ADDR_21:0x[a-z0-9]*]] <<invalid sloc>> Inherited Implicit implementation={vendor(llvm)}
// CXX-NEXT: |   `-DeclRefExpr [[ADDR_9]] <line:10:1> 'int ({{.*}})' Function [[ADDR_10]] 'also_after[implementation={vendor(llvm)}]' 'int ({{.*}})'
// CXX-NEXT: |-FunctionDecl [[ADDR_22:0x[a-z0-9]*]] <line:22:1, col:10> col:6 used foo 'void ({{.*}})'
// CXX-NEXT: |-TypedefDecl [[ADDR_23:0x[a-z0-9]*]] <line:23:1, col:22> col:14 referenced fd 'int (*)({{.*}})'
// CXX-NEXT: | `-PointerType [[ADDR_24:0x[a-z0-9]*]] 'int (*)({{.*}})'
// CXX-NEXT: |   `-ParenType [[ADDR_25:0x[a-z0-9]*]] 'int ({{.*}})' sugar
// CXX-NEXT: |     `-FunctionProtoType [[ADDR_26:0x[a-z0-9]*]] 'int ({{.*}})' cdecl
// CXX-NEXT: |       `-BuiltinType [[ADDR_27:0x[a-z0-9]*]] 'int'
// CXX-NEXT: `-FunctionDecl [[ADDR_28:0x[a-z0-9]*]] <line:24:1, line:32:1> line:24:5 main 'int ({{.*}})'
// CXX-NEXT:   `-CompoundStmt [[ADDR_29:0x[a-z0-9]*]] <col:12, line:32:1>
// CXX-NEXT:     |-DeclStmt [[ADDR_30:0x[a-z0-9]*]] <line:26:3, col:12>
// CXX-NEXT:     | `-VarDecl [[ADDR_31:0x[a-z0-9]*]] <col:3, col:11> col:6 used fns 'fd[2]'
// CXX-NEXT:     |-BinaryOperator [[ADDR_32:0x[a-z0-9]*]] <line:27:3, col:13> 'fd':'int (*)({{.*}})' {{.*}}'='
// CXX-NEXT:     | |-ArraySubscriptExpr [[ADDR_33:0x[a-z0-9]*]] <col:3, col:8> 'fd':'int (*)({{.*}})' lvalue
// CXX-NEXT:     | | |-ImplicitCastExpr [[ADDR_34:0x[a-z0-9]*]] <col:3> 'fd *' <ArrayToPointerDecay>
// CXX-NEXT:     | | | `-DeclRefExpr [[ADDR_35:0x[a-z0-9]*]] <col:3> 'fd[2]' {{.*}}Var [[ADDR_31]] 'fns' 'fd[2]'
// CXX-NEXT:     | | `-IntegerLiteral [[ADDR_36:0x[a-z0-9]*]] <col:7> 'int' 0
// CXX-NEXT:     | `-UnaryOperator [[ADDR_37:0x[a-z0-9]*]] <col:12, col:13> 'int (*)({{.*}})' prefix '&' cannot overflow
// CXX-NEXT:     |   `-DeclRefExpr [[ADDR_38:0x[a-z0-9]*]] <col:13> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'also_before' 'int ({{.*}})'
// CXX-NEXT:     |-BinaryOperator [[ADDR_39:0x[a-z0-9]*]] <line:28:3, col:12> 'fd':'int (*)({{.*}})' {{.*}}'='
// CXX-NEXT:     | |-ArraySubscriptExpr [[ADDR_40:0x[a-z0-9]*]] <col:3, col:8> 'fd':'int (*)({{.*}})' lvalue
// CXX-NEXT:     | | |-ImplicitCastExpr [[ADDR_41:0x[a-z0-9]*]] <col:3> 'fd *' <ArrayToPointerDecay>
// CXX-NEXT:     | | | `-DeclRefExpr [[ADDR_42:0x[a-z0-9]*]] <col:3> 'fd[2]' {{.*}}Var [[ADDR_31]] 'fns' 'fd[2]'
// CXX-NEXT:     | | `-IntegerLiteral [[ADDR_43:0x[a-z0-9]*]] <col:7> 'int' 1
// CXX-NEXT:     | `-ImplicitCastExpr [[ADDR_44:0x[a-z0-9]*]] <col:12> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:     |   `-DeclRefExpr [[ADDR_45:0x[a-z0-9]*]] <col:12> 'int ({{.*}})' {{.*}}Function [[ADDR_17]] 'also_after' 'int ({{.*}})'
// CXX-NEXT:     `-ReturnStmt [[ADDR_46:0x[a-z0-9]*]] <line:29:3, line:31:19>
// CXX-NEXT:       `-BinaryOperator [[ADDR_47:0x[a-z0-9]*]] <line:29:10, line:31:19> 'int' '+'
// CXX-NEXT:         |-BinaryOperator [[ADDR_48:0x[a-z0-9]*]] <line:29:10, line:30:19> 'int' '+'
// CXX-NEXT:         | |-CallExpr [[ADDR_49:0x[a-z0-9]*]] <line:29:10, col:30> 'int'
// CXX-NEXT:         | | `-ImplicitCastExpr [[ADDR_50:0x[a-z0-9]*]] <col:10, col:28> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | |   `-ParenExpr [[ADDR_51:0x[a-z0-9]*]] <col:10, col:28> 'int ({{.*}})' lvalue
// CXX-NEXT:         | |     `-BinaryOperator [[ADDR_52:0x[a-z0-9]*]] <col:11, col:18> 'int ({{.*}})' {{.*}}','
// CXX-NEXT:         | |       |-CallExpr [[ADDR_53:0x[a-z0-9]*]] <col:11, col:15> 'void'
// CXX-NEXT:         | |       | `-ImplicitCastExpr [[ADDR_54:0x[a-z0-9]*]] <col:11> 'void (*)({{.*}})' <FunctionToPointerDecay>
// CXX-NEXT:         | |       |   `-DeclRefExpr [[ADDR_55:0x[a-z0-9]*]] <col:11> 'void ({{.*}})' {{.*}}Function [[ADDR_22]] 'foo' 'void ({{.*}})'
// CXX-NEXT:         | |       `-DeclRefExpr [[ADDR_56:0x[a-z0-9]*]] <col:18> 'int ({{.*}})' {{.*}}Function [[ADDR_17]] 'also_after' 'int ({{.*}})'
// CXX-NEXT:         | `-CallExpr [[ADDR_57:0x[a-z0-9]*]] <line:30:10, col:19> 'int'
// CXX-NEXT:         |   `-ImplicitCastExpr [[ADDR_58:0x[a-z0-9]*]] <col:10, col:17> 'fd':'int (*)({{.*}})' <LValueToRValue>
// CXX-NEXT:         |     `-ParenExpr [[ADDR_59:0x[a-z0-9]*]] <col:10, col:17> 'fd':'int (*)({{.*}})' lvalue
// CXX-NEXT:         |       `-ArraySubscriptExpr [[ADDR_60:0x[a-z0-9]*]] <col:11, col:16> 'fd':'int (*)({{.*}})' lvalue
// CXX-NEXT:         |         |-ImplicitCastExpr [[ADDR_61:0x[a-z0-9]*]] <col:11> 'fd *' <ArrayToPointerDecay>
// CXX-NEXT:         |         | `-DeclRefExpr [[ADDR_62:0x[a-z0-9]*]] <col:11> 'fd[2]' {{.*}}Var [[ADDR_31]] 'fns' 'fd[2]'
// CXX-NEXT:         |         `-IntegerLiteral [[ADDR_63:0x[a-z0-9]*]] <col:15> 'int' 0
// CXX-NEXT:         `-CallExpr [[ADDR_64:0x[a-z0-9]*]] <line:31:10, col:19> 'int'
// CXX-NEXT:           `-ImplicitCastExpr [[ADDR_65:0x[a-z0-9]*]] <col:10, col:17> 'fd':'int (*)({{.*}})' <LValueToRValue>
// CXX-NEXT:             `-ParenExpr [[ADDR_66:0x[a-z0-9]*]] <col:10, col:17> 'fd':'int (*)({{.*}})' lvalue
// CXX-NEXT:               `-ArraySubscriptExpr [[ADDR_67:0x[a-z0-9]*]] <col:11, col:16> 'fd':'int (*)({{.*}})' lvalue
// CXX-NEXT:                 |-IntegerLiteral [[ADDR_68:0x[a-z0-9]*]] <col:11> 'int' 1
// CXX-NEXT:                 `-ImplicitCastExpr [[ADDR_69:0x[a-z0-9]*]] <col:13> 'fd *' <ArrayToPointerDecay>
// CXX-NEXT:                   `-DeclRefExpr [[ADDR_70:0x[a-z0-9]*]] <col:13> 'fd[2]' {{.*}}Var [[ADDR_31]] 'fns' 'fd[2]'
