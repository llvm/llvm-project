// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test_one(int x) {
#pragma omp target
#pragma omp teams distribute
  for (int i = 0; i < x; i++)
    ;
}

void test_two(int x, int y) {
#pragma omp target
#pragma omp teams distribute
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_three(int x, int y) {
#pragma omp target
#pragma omp teams distribute collapse(1)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_four(int x, int y) {
#pragma omp target
#pragma omp teams distribute collapse(2)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_five(int x, int y, int z) {
#pragma omp target
#pragma omp teams distribute collapse(2)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      for (int i = 0; i < z; i++)
        ;
}
// CHECK: TranslationUnitDecl 0x{{.+}} <<invalid sloc>> <invalid sloc>
// CHECK: |-TypedefDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> implicit __int128_t '__int128'
// CHECK: | `-typeDetails: BuiltinType 0x{{.+}} '__int128'
// CHECK: |-TypedefDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> implicit __uint128_t 'unsigned __int128'
// CHECK: | `-typeDetails: BuiltinType 0x{{.+}} 'unsigned __int128'
// CHECK: |-TypedefDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> implicit __NSConstantString 'struct __NSConstantString_tag'
// CHECK: | `-typeDetails: RecordType 0x{{.+}} 'struct __NSConstantString_tag'
// CHECK: |   `-Record 0x{{.+}} '__NSConstantString_tag'
// CHECK: |-TypedefDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> implicit __builtin_ms_va_list 'char *'
// CHECK: | `-typeDetails: PointerType 0x{{.+}} 'char *'
// CHECK: |   `-typeDetails: BuiltinType 0x{{.+}} 'char'
// CHECK: |-TypedefDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> implicit __builtin_va_list 'struct __va_list_tag[1]'
// CHECK: | `-typeDetails: ConstantArrayType 0x{{.+}} 'struct __va_list_tag[1]' 1
// CHECK: |   `-typeDetails: RecordType 0x{{.+}} 'struct __va_list_tag'
// CHECK: |     `-Record 0x{{.+}} '__va_list_tag'
// CHECK: |-FunctionDecl 0x{{.+}} <{{.*}} line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test_one 'void (int)'
// CHECK: | |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used x 'int'
// CHECK: | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |   `-OMPTargetDirective 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |     |-OMPFirstprivateClause 0x{{.+}} <<invalid sloc>> <implicit>
// CHECK: |     | `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |     `-CapturedStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | |-CapturedStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK: |       | | |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |-OMPTeamsDistributeDirective 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK: |       | | | | `-CapturedStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | |   |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | |   | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       | | | |   | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | | |-<<<NULL>>>
// CHECK: |       | | | |   | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       | | | |   | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | |   | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       | | | |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | |   | | `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK: |       | | | |   | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | | |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | | |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       | | | |   | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       | | | |   | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       | | | |   | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | |   | | |     `-Record 0x{{.+}} ''
// CHECK: |       | | | |   | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | |   | |   `-Record 0x{{.+}} ''
// CHECK: |       | | | |   | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} 
// CHECK: |       | | | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       | | | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       | | | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | | |     `-Record 0x{{.+}} ''
// CHECK: |       | | | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | |   `-Record 0x{{.+}} ''
// CHECK: |       | | | |-RecordDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit struct definition
// CHECK: |       | | | | |-attrDetails: CapturedRecordAttr 0x{{.+}} <<invalid sloc>> Implicit
// CHECK: |       | | | | `-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK: |       | | | |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | | |-ForStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       | | | | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | | |-<<<NULL>>>
// CHECK: |       | | | | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       | | | | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       | | | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | | | `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK: |       | | | | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | | | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | | | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       | | | | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       | | | | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       | | | | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | | | |     `-Record 0x{{.+}} ''
// CHECK: |       | | | | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | | |   `-Record 0x{{.+}} ''
// CHECK: |       | | | | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |-OMPCapturedExprDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK: |       | | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | `-OMPCapturedExprDecl 0x{{.+}} <col:{{.*}}, <invalid sloc>> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK: |       | | |   `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '-'
// CHECK: |       | | |     |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '/'
// CHECK: |       | | |     | |-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       | | |     | | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       | | |     | |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | |     | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue OMPCapturedExpr 0x{{.+}} '.capture_expr.' 'int'
// CHECK: |       | | |     | |   `-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       | | |     | |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '+'
// CHECK: |       | | |     | |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       | | |     | |       | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | |     | |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       | | |     | |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       | | |     | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       | | |     `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | |-attrDetails: AlwaysInlineAttr 0x{{.+}} <<invalid sloc>> Implicit __forceinline
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int'
// CHECK: |       | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .part_id. 'const int *const restrict'
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .privates. 'void *const restrict'
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 'void *const restrict' const __restrict
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK: |       | | |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 'void (*const restrict)(void *const restrict, ...)' const __restrict
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 'void (*)(void *const restrict, ...)'
// CHECK: |       | | |   `-typeDetails: FunctionProtoType 0x{{.+}} 'void (void *const restrict, ...)' variadic cdecl
// CHECK: |       | | |     |-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | |     |-functionDetails:  cdeclReturnType 0x{{.+}} 'void'
// CHECK: |       | | |     |-qualTypeDetail: QualType 0x{{.+}} 'void *const restrict' const __restrict
// CHECK: |       | | |     | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK: |       | | |     |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | |     `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | |-QualType 0x{{.+}} 'void (void *const restrict, ...)'
// CHECK: |       | | `-typeDetails: FunctionProtoType 0x{{.+}} 'void (void *const restrict, ...)' variadic cdecl
// CHECK: |       | |   |-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |   |-functionDetails:  cdeclReturnType 0x{{.+}} 'void'
// CHECK: |       | |   |-qualTypeDetail: QualType 0x{{.+}} 'void *const restrict' const __restrict
// CHECK: |       | |   | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK: |       | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .task_t. 'void *const'
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 'void *const' const
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK: |       | | |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | |     `-Record 0x{{.+}} ''
// CHECK: |       | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | |   `-Record 0x{{.+}} ''
// CHECK: |       | |-RecordDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit struct definition
// CHECK: |       | | |-attrDetails: CapturedRecordAttr 0x{{.+}} <<invalid sloc>> Implicit
// CHECK: |       | | `-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int'
// CHECK: |       | |   `-attrDetails: OMPCaptureKindAttr 0x{{.+}} <<invalid sloc>> Implicit 36
// CHECK: |       | `-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |-OMPTeamsDistributeDirective 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   | `-CapturedStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   |   |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   |   | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   |   | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | | |-<<<NULL>>>
// CHECK: |       |   |   | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       |   |   | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   |   | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       |   |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   |   | | `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK: |       |   |   | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       |   |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       |   |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       |   |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       |   |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       |   |   | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       |   |   | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       |   |   | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   |   | | |     `-Record 0x{{.+}} ''
// CHECK: |       |   |   | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   |   | |   `-Record 0x{{.+}} ''
// CHECK: |       |   |   | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} 
// CHECK: |       |   | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       |   | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       |   | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   | |     `-Record 0x{{.+}} ''
// CHECK: |       |   | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   |   `-Record 0x{{.+}} ''
// CHECK: |       |   |-RecordDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit struct definition
// CHECK: |       |   | |-attrDetails: CapturedRecordAttr 0x{{.+}} <<invalid sloc>> Implicit
// CHECK: |       |   | `-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK: |       |   |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   | |-ForStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | | |-<<<NULL>>>
// CHECK: |       |   | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       |   | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   | | `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK: |       |   | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       |   | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       |   | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       |   | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   | | |     `-Record 0x{{.+}} ''
// CHECK: |       |   | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   | |   `-Record 0x{{.+}} ''
// CHECK: |       |   | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |-OMPCapturedExprDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK: |       |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   `-OMPCapturedExprDecl 0x{{.+}} <col:{{.*}}, <invalid sloc>> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK: |       |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '-'
// CHECK: |       |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '/'
// CHECK: |       |       | |-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       |       | | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       |       | |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |       | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue OMPCapturedExpr 0x{{.+}} '.capture_expr.' 'int'
// CHECK: |       |       | |   `-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       |       | |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '+'
// CHECK: |       |       | |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       |       | |       | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |       | |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       |       | |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int'
// CHECK: |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test_two 'void (int, int)'
// CHECK: | |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used x 'int'
// CHECK: | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: | |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used y 'int'
// CHECK: | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |   `-OMPTargetDirective 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |     |-OMPFirstprivateClause 0x{{.+}} <<invalid sloc>> <implicit>
// CHECK: |     | |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |     | `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |     `-CapturedStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | |-CapturedStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK: |       | | |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |-OMPTeamsDistributeDirective 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK: |       | | | | `-CapturedStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | |   |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | |   | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       | | | |   | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | | |-<<<NULL>>>
// CHECK: |       | | | |   | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       | | | |   | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | |   | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       | | | |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | |   | | `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | |   | |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       | | | |   | |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | |   |-<<<NULL>>>
// CHECK: |       | | | |   | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       | | | |   | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | |   | |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | |   |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       | | | |   | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | |   | |   `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK: |       | | | |   | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | | |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | | |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       | | | |   | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       | | | |   | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       | | | |   | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | |   | | |     `-Record 0x{{.+}} ''
// CHECK: |       | | | |   | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | |   | |   `-Record 0x{{.+}} ''
// CHECK: |       | | | |   | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} 
// CHECK: |       | | | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       | | | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       | | | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | | |     `-Record 0x{{.+}} ''
// CHECK: |       | | | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | |   `-Record 0x{{.+}} ''
// CHECK: |       | | | |-RecordDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit struct definition
// CHECK: |       | | | | |-attrDetails: CapturedRecordAttr 0x{{.+}} <<invalid sloc>> Implicit
// CHECK: |       | | | | |-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK: |       | | | | `-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK: |       | | | |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       | | | | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | | |-<<<NULL>>>
// CHECK: |       | | | | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       | | | | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       | | | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | | | `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | | |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       | | | | |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | |   |-<<<NULL>>>
// CHECK: |       | | | | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       | | | | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | | |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | |   |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       | | | | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | | |   `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK: |       | | | | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | | | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | | | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       | | | | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       | | | | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       | | | | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | | | |     `-Record 0x{{.+}} ''
// CHECK: |       | | | | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | | |   `-Record 0x{{.+}} ''
// CHECK: |       | | | | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |-OMPCapturedExprDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK: |       | | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | `-OMPCapturedExprDecl 0x{{.+}} <col:{{.*}}, <invalid sloc>> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK: |       | | |   `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '-'
// CHECK: |       | | |     |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '/'
// CHECK: |       | | |     | |-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       | | |     | | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       | | |     | |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | |     | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue OMPCapturedExpr 0x{{.+}} '.capture_expr.' 'int'
// CHECK: |       | | |     | |   `-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       | | |     | |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '+'
// CHECK: |       | | |     | |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       | | |     | |       | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | |     | |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       | | |     | |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       | | |     | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       | | |     `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       | | |-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | |-attrDetails: AlwaysInlineAttr 0x{{.+}} <<invalid sloc>> Implicit __forceinline
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int'
// CHECK: |       | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .part_id. 'const int *const restrict'
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .privates. 'void *const restrict'
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 'void *const restrict' const __restrict
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK: |       | | |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 'void (*const restrict)(void *const restrict, ...)' const __restrict
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 'void (*)(void *const restrict, ...)'
// CHECK: |       | | |   `-typeDetails: FunctionProtoType 0x{{.+}} 'void (void *const restrict, ...)' variadic cdecl
// CHECK: |       | | |     |-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | |     |-functionDetails:  cdeclReturnType 0x{{.+}} 'void'
// CHECK: |       | | |     |-qualTypeDetail: QualType 0x{{.+}} 'void *const restrict' const __restrict
// CHECK: |       | | |     | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK: |       | | |     |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | |     `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | |-QualType 0x{{.+}} 'void (void *const restrict, ...)'
// CHECK: |       | | `-typeDetails: FunctionProtoType 0x{{.+}} 'void (void *const restrict, ...)' variadic cdecl
// CHECK: |       | |   |-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |   |-functionDetails:  cdeclReturnType 0x{{.+}} 'void'
// CHECK: |       | |   |-qualTypeDetail: QualType 0x{{.+}} 'void *const restrict' const __restrict
// CHECK: |       | |   | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK: |       | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .task_t. 'void *const'
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 'void *const' const
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK: |       | | |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | |     `-Record 0x{{.+}} ''
// CHECK: |       | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | |   `-Record 0x{{.+}} ''
// CHECK: |       | |-RecordDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit struct definition
// CHECK: |       | | |-attrDetails: CapturedRecordAttr 0x{{.+}} <<invalid sloc>> Implicit
// CHECK: |       | | |-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int'
// CHECK: |       | | | `-attrDetails: OMPCaptureKindAttr 0x{{.+}} <<invalid sloc>> Implicit 36
// CHECK: |       | | `-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int'
// CHECK: |       | |   `-attrDetails: OMPCaptureKindAttr 0x{{.+}} <<invalid sloc>> Implicit 36
// CHECK: |       | `-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |-OMPTeamsDistributeDirective 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   | `-CapturedStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   |   |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   |   | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   |   | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | | |-<<<NULL>>>
// CHECK: |       |   |   | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       |   |   | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   |   | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       |   |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   |   | | `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   |   | |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   |   | |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | |   |-<<<NULL>>>
// CHECK: |       |   |   | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       |   |   | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   |   | |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | |   |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       |   |   | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   |   | |   `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK: |       |   |   | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       |   |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       |   |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       |   |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       |   |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       |   |   | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       |   |   | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       |   |   | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   |   | | |     `-Record 0x{{.+}} ''
// CHECK: |       |   |   | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   |   | |   `-Record 0x{{.+}} ''
// CHECK: |       |   |   | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} 
// CHECK: |       |   | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       |   | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       |   | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   | |     `-Record 0x{{.+}} ''
// CHECK: |       |   | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   |   `-Record 0x{{.+}} ''
// CHECK: |       |   |-RecordDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit struct definition
// CHECK: |       |   | |-attrDetails: CapturedRecordAttr 0x{{.+}} <<invalid sloc>> Implicit
// CHECK: |       |   | |-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK: |       |   | `-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK: |       |   |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | | |-<<<NULL>>>
// CHECK: |       |   | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       |   | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   | | `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   | |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   | |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | |   |-<<<NULL>>>
// CHECK: |       |   | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       |   | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   | |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | |   |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       |   | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   | |   `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK: |       |   | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       |   | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       |   | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       |   | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   | | |     `-Record 0x{{.+}} ''
// CHECK: |       |   | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   | |   `-Record 0x{{.+}} ''
// CHECK: |       |   | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |-OMPCapturedExprDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK: |       |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   `-OMPCapturedExprDecl 0x{{.+}} <col:{{.*}}, <invalid sloc>> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK: |       |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '-'
// CHECK: |       |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '/'
// CHECK: |       |       | |-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       |       | | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       |       | |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |       | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue OMPCapturedExpr 0x{{.+}} '.capture_expr.' 'int'
// CHECK: |       |       | |   `-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       |       | |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '+'
// CHECK: |       |       | |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       |       | |       | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |       | |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       |       | |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       |-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int'
// CHECK: |       `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int'
// CHECK: |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test_three 'void (int, int)'
// CHECK: | |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used x 'int'
// CHECK: | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: | |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used y 'int'
// CHECK: | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |   `-OMPTargetDirective 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |     |-OMPFirstprivateClause 0x{{.+}} <<invalid sloc>> <implicit>
// CHECK: |     | |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |     | `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |     `-CapturedStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | |-CapturedStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK: |       | | |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |-OMPTeamsDistributeDirective 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK: |       | | | | |-OMPCollapseClause 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK: |       | | | | | `-ConstantExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       | | | | |   |-value: Int 1
// CHECK: |       | | | | |   `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       | | | | `-CapturedStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | |   |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | |   | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       | | | |   | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | | |-<<<NULL>>>
// CHECK: |       | | | |   | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       | | | |   | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | |   | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       | | | |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | |   | | `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | |   | |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       | | | |   | |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | |   |-<<<NULL>>>
// CHECK: |       | | | |   | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       | | | |   | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | |   | |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | |   |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       | | | |   | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | |   | |   `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK: |       | | | |   | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | | |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | | |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       | | | |   | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       | | | |   | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       | | | |   | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | |   | | |     `-Record 0x{{.+}} ''
// CHECK: |       | | | |   | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | |   | |   `-Record 0x{{.+}} ''
// CHECK: |       | | | |   | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} 
// CHECK: |       | | | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       | | | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       | | | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | | |     `-Record 0x{{.+}} ''
// CHECK: |       | | | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | |   `-Record 0x{{.+}} ''
// CHECK: |       | | | |-RecordDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit struct definition
// CHECK: |       | | | | |-attrDetails: CapturedRecordAttr 0x{{.+}} <<invalid sloc>> Implicit
// CHECK: |       | | | | |-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK: |       | | | | `-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK: |       | | | |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       | | | | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | | |-<<<NULL>>>
// CHECK: |       | | | | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       | | | | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       | | | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | | | `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | | |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       | | | | |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | |   |-<<<NULL>>>
// CHECK: |       | | | | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       | | | | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | | |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | |   |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       | | | | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | | |   `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK: |       | | | | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | | | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | | | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       | | | | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       | | | | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       | | | | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | | | |     `-Record 0x{{.+}} ''
// CHECK: |       | | | | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | | |   `-Record 0x{{.+}} ''
// CHECK: |       | | | | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |-OMPCapturedExprDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK: |       | | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | `-OMPCapturedExprDecl 0x{{.+}} <col:{{.*}}, <invalid sloc>> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK: |       | | |   `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '-'
// CHECK: |       | | |     |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '/'
// CHECK: |       | | |     | |-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       | | |     | | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       | | |     | |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | |     | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue OMPCapturedExpr 0x{{.+}} '.capture_expr.' 'int'
// CHECK: |       | | |     | |   `-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       | | |     | |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '+'
// CHECK: |       | | |     | |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       | | |     | |       | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | |     | |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       | | |     | |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       | | |     | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       | | |     `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       | | |-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | |-attrDetails: AlwaysInlineAttr 0x{{.+}} <<invalid sloc>> Implicit __forceinline
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int'
// CHECK: |       | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .part_id. 'const int *const restrict'
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .privates. 'void *const restrict'
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 'void *const restrict' const __restrict
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK: |       | | |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 'void (*const restrict)(void *const restrict, ...)' const __restrict
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 'void (*)(void *const restrict, ...)'
// CHECK: |       | | |   `-typeDetails: FunctionProtoType 0x{{.+}} 'void (void *const restrict, ...)' variadic cdecl
// CHECK: |       | | |     |-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | |     |-functionDetails:  cdeclReturnType 0x{{.+}} 'void'
// CHECK: |       | | |     |-qualTypeDetail: QualType 0x{{.+}} 'void *const restrict' const __restrict
// CHECK: |       | | |     | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK: |       | | |     |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | |     `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | |-QualType 0x{{.+}} 'void (void *const restrict, ...)'
// CHECK: |       | | `-typeDetails: FunctionProtoType 0x{{.+}} 'void (void *const restrict, ...)' variadic cdecl
// CHECK: |       | |   |-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |   |-functionDetails:  cdeclReturnType 0x{{.+}} 'void'
// CHECK: |       | |   |-qualTypeDetail: QualType 0x{{.+}} 'void *const restrict' const __restrict
// CHECK: |       | |   | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK: |       | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .task_t. 'void *const'
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 'void *const' const
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK: |       | | |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | |     `-Record 0x{{.+}} ''
// CHECK: |       | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | |   `-Record 0x{{.+}} ''
// CHECK: |       | |-RecordDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit struct definition
// CHECK: |       | | |-attrDetails: CapturedRecordAttr 0x{{.+}} <<invalid sloc>> Implicit
// CHECK: |       | | |-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int'
// CHECK: |       | | | `-attrDetails: OMPCaptureKindAttr 0x{{.+}} <<invalid sloc>> Implicit 36
// CHECK: |       | | `-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int'
// CHECK: |       | |   `-attrDetails: OMPCaptureKindAttr 0x{{.+}} <<invalid sloc>> Implicit 36
// CHECK: |       | `-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |-OMPTeamsDistributeDirective 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   | |-OMPCollapseClause 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK: |       |   | | `-ConstantExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       |   | |   |-value: Int 1
// CHECK: |       |   | |   `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       |   | `-CapturedStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   |   |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   |   | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   |   | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | | |-<<<NULL>>>
// CHECK: |       |   |   | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       |   |   | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   |   | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       |   |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   |   | | `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   |   | |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   |   | |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | |   |-<<<NULL>>>
// CHECK: |       |   |   | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       |   |   | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   |   | |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | |   |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       |   |   | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   |   | |   `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK: |       |   |   | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       |   |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       |   |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       |   |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       |   |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       |   |   | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       |   |   | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       |   |   | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   |   | | |     `-Record 0x{{.+}} ''
// CHECK: |       |   |   | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   |   | |   `-Record 0x{{.+}} ''
// CHECK: |       |   |   | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} 
// CHECK: |       |   | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       |   | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       |   | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   | |     `-Record 0x{{.+}} ''
// CHECK: |       |   | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   |   `-Record 0x{{.+}} ''
// CHECK: |       |   |-RecordDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit struct definition
// CHECK: |       |   | |-attrDetails: CapturedRecordAttr 0x{{.+}} <<invalid sloc>> Implicit
// CHECK: |       |   | |-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK: |       |   | `-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK: |       |   |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | | |-<<<NULL>>>
// CHECK: |       |   | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       |   | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   | | `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   | |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   | |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | |   |-<<<NULL>>>
// CHECK: |       |   | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       |   | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   | |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | |   |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       |   | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   | |   `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK: |       |   | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       |   | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       |   | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       |   | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   | | |     `-Record 0x{{.+}} ''
// CHECK: |       |   | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   | |   `-Record 0x{{.+}} ''
// CHECK: |       |   | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |-OMPCapturedExprDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK: |       |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   `-OMPCapturedExprDecl 0x{{.+}} <col:{{.*}}, <invalid sloc>> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK: |       |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '-'
// CHECK: |       |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '/'
// CHECK: |       |       | |-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       |       | | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       |       | |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |       | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue OMPCapturedExpr 0x{{.+}} '.capture_expr.' 'int'
// CHECK: |       |       | |   `-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       |       | |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '+'
// CHECK: |       |       | |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       |       | |       | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |       | |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       |       | |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       |-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int'
// CHECK: |       `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int'
// CHECK: |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test_four 'void (int, int)'
// CHECK: | |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used x 'int'
// CHECK: | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: | |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used y 'int'
// CHECK: | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |   `-OMPTargetDirective 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |     |-OMPFirstprivateClause 0x{{.+}} <<invalid sloc>> <implicit>
// CHECK: |     | |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |     | `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |     `-CapturedStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | |-CapturedStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK: |       | | |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |-OMPTeamsDistributeDirective 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK: |       | | | | |-OMPCollapseClause 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK: |       | | | | | `-ConstantExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       | | | | |   |-value: Int 2
// CHECK: |       | | | | |   `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 2
// CHECK: |       | | | | `-CapturedStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | |   |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | |   | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | |   | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       | | | |   | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | | |-<<<NULL>>>
// CHECK: |       | | | |   | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       | | | |   | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | |   | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       | | | |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | |   | | `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | |   | |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       | | | |   | |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | |   |-<<<NULL>>>
// CHECK: |       | | | |   | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       | | | |   | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | |   | |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   | |   |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       | | | |   | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | |   | |   `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK: |       | | | |   | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | | |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | | |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       | | | |   | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       | | | |   | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       | | | |   | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | |   | | |     `-Record 0x{{.+}} ''
// CHECK: |       | | | |   | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | |   | |   `-Record 0x{{.+}} ''
// CHECK: |       | | | |   | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |   |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |   `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} 
// CHECK: |       | | | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       | | | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       | | | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | | |     `-Record 0x{{.+}} ''
// CHECK: |       | | | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | |   `-Record 0x{{.+}} ''
// CHECK: |       | | | |-RecordDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit struct definition
// CHECK: |       | | | | |-attrDetails: CapturedRecordAttr 0x{{.+}} <<invalid sloc>> Implicit
// CHECK: |       | | | | |-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK: |       | | | | `-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK: |       | | | |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       | | | | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       | | | | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | | |-<<<NULL>>>
// CHECK: |       | | | | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       | | | | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       | | | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | | | `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       | | | | |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       | | | | |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | |   |-<<<NULL>>>
// CHECK: |       | | | | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       | | | | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | | |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | | |   |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       | | | | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       | | | | |   `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK: |       | | | | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK: |       | | | | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | | | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK: |       | | | | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | | | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       | | | | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       | | | | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       | | | | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | | | |     `-Record 0x{{.+}} ''
// CHECK: |       | | | | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | | | |   `-Record 0x{{.+}} ''
// CHECK: |       | | | | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       | | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | | |-OMPCapturedExprDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK: |       | | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | |-OMPCapturedExprDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK: |       | | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | | `-OMPCapturedExprDecl 0x{{.+}} <line:{{.*}}:{{.*}}, <invalid sloc>> col:{{.*}} implicit used .capture_expr. 'long'
// CHECK: |       | | |   `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'long' '-'
// CHECK: |       | | |     |-BinaryOperator 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}> 'long' '*'
// CHECK: |       | | |     | |-ImplicitCastExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'long' <IntegralCast>
// CHECK: |       | | |     | | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '/'
// CHECK: |       | | |     | |   |-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       | | |     | |   | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       | | |     | |   |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | |     | |   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue OMPCapturedExpr 0x{{.+}} '.capture_expr.' 'int'
// CHECK: |       | | |     | |   |   `-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       | | |     | |   |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '+'
// CHECK: |       | | |     | |   |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       | | |     | |   |       | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | |     | |   |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       | | |     | |   |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       | | |     | |   `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       | | |     | `-ImplicitCastExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'long' <IntegralCast>
// CHECK: |       | | |     |   `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '/'
// CHECK: |       | | |     |     |-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       | | |     |     | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       | | |     |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       | | |     |     |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue OMPCapturedExpr 0x{{.+}} '.capture_expr.' 'int'
// CHECK: |       | | |     |     |   `-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       | | |     |     |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '+'
// CHECK: |       | | |     |     |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       | | |     |     |       | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       | | |     |     |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       | | |     |     |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       | | |     |     `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       | | |     `-ImplicitCastExpr 0x{{.+}} <<invalid sloc>> 'long' <IntegralCast>
// CHECK: |       | | |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       | | |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | | `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       | |-attrDetails: AlwaysInlineAttr 0x{{.+}} <<invalid sloc>> Implicit __forceinline
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int'
// CHECK: |       | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .part_id. 'const int *const restrict'
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .privates. 'void *const restrict'
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 'void *const restrict' const __restrict
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK: |       | | |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 'void (*const restrict)(void *const restrict, ...)' const __restrict
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 'void (*)(void *const restrict, ...)'
// CHECK: |       | | |   `-typeDetails: FunctionProtoType 0x{{.+}} 'void (void *const restrict, ...)' variadic cdecl
// CHECK: |       | | |     |-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | |     |-functionDetails:  cdeclReturnType 0x{{.+}} 'void'
// CHECK: |       | | |     |-qualTypeDetail: QualType 0x{{.+}} 'void *const restrict' const __restrict
// CHECK: |       | | |     | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK: |       | | |     |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | |     `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | |-QualType 0x{{.+}} 'void (void *const restrict, ...)'
// CHECK: |       | | `-typeDetails: FunctionProtoType 0x{{.+}} 'void (void *const restrict, ...)' variadic cdecl
// CHECK: |       | |   |-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |   |-functionDetails:  cdeclReturnType 0x{{.+}} 'void'
// CHECK: |       | |   |-qualTypeDetail: QualType 0x{{.+}} 'void *const restrict' const __restrict
// CHECK: |       | |   | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK: |       | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .task_t. 'void *const'
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 'void *const' const
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK: |       | | |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | | `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK: |       | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | | |     `-Record 0x{{.+}} ''
// CHECK: |       | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       | |   `-Record 0x{{.+}} ''
// CHECK: |       | |-RecordDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit struct definition
// CHECK: |       | | |-attrDetails: CapturedRecordAttr 0x{{.+}} <<invalid sloc>> Implicit
// CHECK: |       | | |-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int'
// CHECK: |       | | | `-attrDetails: OMPCaptureKindAttr 0x{{.+}} <<invalid sloc>> Implicit 36
// CHECK: |       | | `-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int'
// CHECK: |       | |   `-attrDetails: OMPCaptureKindAttr 0x{{.+}} <<invalid sloc>> Implicit 36
// CHECK: |       | `-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |-OMPTeamsDistributeDirective 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   | |-OMPCollapseClause 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK: |       |   | | `-ConstantExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       |   | |   |-value: Int 2
// CHECK: |       |   | |   `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 2
// CHECK: |       |   | `-CapturedStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   |   |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   |   | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   |   | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   |   | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | | |-<<<NULL>>>
// CHECK: |       |   |   | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       |   |   | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   |   | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       |   |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   |   | | `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   |   | |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   |   | |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | |   |-<<<NULL>>>
// CHECK: |       |   |   | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       |   |   | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   |   | |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   | |   |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       |   |   | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   |   | |   `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK: |       |   |   | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       |   |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       |   |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       |   |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       |   |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       |   |   | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       |   |   | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       |   |   | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   |   | | |     `-Record 0x{{.+}} ''
// CHECK: |       |   |   | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   |   | |   `-Record 0x{{.+}} ''
// CHECK: |       |   |   | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |   |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |   `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} 
// CHECK: |       |   | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       |   | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       |   | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   | |     `-Record 0x{{.+}} ''
// CHECK: |       |   | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   |   `-Record 0x{{.+}} ''
// CHECK: |       |   |-RecordDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit struct definition
// CHECK: |       |   | |-attrDetails: CapturedRecordAttr 0x{{.+}} <<invalid sloc>> Implicit
// CHECK: |       |   | |-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK: |       |   | `-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK: |       |   |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK: |       |   | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | | |-<<<NULL>>>
// CHECK: |       |   | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       |   | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   | | `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK: |       |   | |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK: |       |   | |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | |   |-<<<NULL>>>
// CHECK: |       |   | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK: |       |   | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   | |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   | |   |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK: |       |   | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK: |       |   | |   `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK: |       |   | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK: |       |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK: |       |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK: |       |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK: |       |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK: |       |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK: |       |   | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK: |       |   | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK: |       |   | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   | | |     `-Record 0x{{.+}} ''
// CHECK: |       |   | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK: |       |   | |   `-Record 0x{{.+}} ''
// CHECK: |       |   | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK: |       |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK: |       |   |-OMPCapturedExprDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK: |       |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   |-OMPCapturedExprDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK: |       |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK: |       |   `-OMPCapturedExprDecl 0x{{.+}} <line:{{.*}}:{{.*}}, <invalid sloc>> col:{{.*}} implicit used .capture_expr. 'long'
// CHECK: |       |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'long' '-'
// CHECK: |       |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}> 'long' '*'
// CHECK: |       |       | |-ImplicitCastExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'long' <IntegralCast>
// CHECK: |       |       | | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '/'
// CHECK: |       |       | |   |-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       |       | |   | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       |       | |   |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |       | |   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue OMPCapturedExpr 0x{{.+}} '.capture_expr.' 'int'
// CHECK: |       |       | |   |   `-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       |       | |   |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '+'
// CHECK: |       |       | |   |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       |       | |   |       | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |       | |   |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       |       | |   |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       |       | |   `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       |       | `-ImplicitCastExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'long' <IntegralCast>
// CHECK: |       |       |   `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '/'
// CHECK: |       |       |     |-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       |       |     | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       |       |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK: |       |       |     |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue OMPCapturedExpr 0x{{.+}} '.capture_expr.' 'int'
// CHECK: |       |       |     |   `-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK: |       |       |     |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '+'
// CHECK: |       |       |     |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK: |       |       |     |       | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK: |       |       |     |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       |       |     |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       |       |     `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK: |       |       `-ImplicitCastExpr 0x{{.+}} <<invalid sloc>> 'long' <IntegralCast>
// CHECK: |       |         `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK: |       |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int'
// CHECK: |       `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int'
// CHECK: `-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test_five 'void (int, int, int)'
// CHECK:   |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used x 'int'
// CHECK:   | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used y 'int'
// CHECK:   | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used z 'int'
// CHECK:   | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:     `-OMPTargetDirective 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:       |-OMPFirstprivateClause 0x{{.+}} <<invalid sloc>> <implicit>
// CHECK:       | |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:       | |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:       | `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:       `-CapturedStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:         |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         | |-CapturedStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:         | | |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         | | | |-OMPTeamsDistributeDirective 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:         | | | | |-OMPCollapseClause 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:         | | | | | `-ConstantExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK:         | | | | |   |-value: Int 2
// CHECK:         | | | | |   `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 2
// CHECK:         | | | | `-CapturedStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:         | | | |   |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         | | | |   | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:         | | | |   | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:         | | | |   | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         | | | |   | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         | | | |   | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | |   | | |-<<<NULL>>>
// CHECK:         | | | |   | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK:         | | | |   | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         | | | |   | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         | | | |   | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         | | | |   | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK:         | | | |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         | | | |   | | `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:         | | | |   | |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:         | | | |   | |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         | | | |   | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         | | | |   | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | |   | |   |-<<<NULL>>>
// CHECK:         | | | |   | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK:         | | | |   | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         | | | |   | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         | | | |   | |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         | | | |   | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   | |   |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK:         | | | |   | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         | | | |   | |   `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:         | | | |   | |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:         | | | |   | |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         | | | |   | |     |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         | | | |   | |     |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | |   | |     |-<<<NULL>>>
// CHECK:         | | | |   | |     |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK:         | | | |   | |     | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         | | | |   | |     | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         | | | |   | |     | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         | | | |   | |     |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   | |     |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK:         | | | |   | |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         | | | |   | |     `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:         | | | |   | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK:         | | | |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK:         | | | |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:         | | | |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         | | | |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         | | | |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK:         | | | |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK:         | | | |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:         | | | |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         | | | |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         | | | |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK:         | | | |   | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK:         | | | |   | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK:         | | | |   | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK:         | | | |   | | |     `-Record 0x{{.+}} ''
// CHECK:         | | | |   | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK:         | | | |   | |   `-Record 0x{{.+}} ''
// CHECK:         | | | |   | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         | | | |   | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         | | | |   | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | |   | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         | | | |   | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         | | | |   | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | |   | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         | | | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         | | | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | |   |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |   `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} 
// CHECK:         | | | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK:         | | | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK:         | | | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK:         | | | | |     `-Record 0x{{.+}} ''
// CHECK:         | | | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK:         | | | |   `-Record 0x{{.+}} ''
// CHECK:         | | | |-RecordDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit struct definition
// CHECK:         | | | | |-attrDetails: CapturedRecordAttr 0x{{.+}} <<invalid sloc>> Implicit
// CHECK:         | | | | |-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK:         | | | | |-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK:         | | | | `-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK:         | | | |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         | | | | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:         | | | | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:         | | | | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         | | | | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         | | | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | | | |-<<<NULL>>>
// CHECK:         | | | | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK:         | | | | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         | | | | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         | | | | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         | | | | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK:         | | | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         | | | | | `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:         | | | | |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:         | | | | |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         | | | | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         | | | | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | | |   |-<<<NULL>>>
// CHECK:         | | | | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK:         | | | | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         | | | | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         | | | | |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         | | | | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | | |   |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK:         | | | | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         | | | | |   `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:         | | | | |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:         | | | | |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         | | | | |     |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         | | | | |     |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | | |     |-<<<NULL>>>
// CHECK:         | | | | |     |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK:         | | | | |     | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         | | | | |     | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         | | | | |     | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         | | | | |     |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | | |     |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK:         | | | | |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         | | | | |     `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:         | | | | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK:         | | | | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK:         | | | | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:         | | | | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         | | | | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         | | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK:         | | | | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK:         | | | | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:         | | | | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         | | | | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         | | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK:         | | | | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK:         | | | | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK:         | | | | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK:         | | | | | |     `-Record 0x{{.+}} ''
// CHECK:         | | | | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK:         | | | | |   `-Record 0x{{.+}} ''
// CHECK:         | | | | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         | | | | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         | | | | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         | | | | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         | | | | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         | | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         | | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | | |-OMPCapturedExprDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK:         | | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         | | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | |-OMPCapturedExprDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK:         | | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         | | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | | `-OMPCapturedExprDecl 0x{{.+}} <line:{{.*}}:{{.*}}, <invalid sloc>> col:{{.*}} implicit used .capture_expr. 'long'
// CHECK:         | | |   `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'long' '-'
// CHECK:         | | |     |-BinaryOperator 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}> 'long' '*'
// CHECK:         | | |     | |-ImplicitCastExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'long' <IntegralCast>
// CHECK:         | | |     | | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '/'
// CHECK:         | | |     | |   |-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK:         | | |     | |   | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK:         | | |     | |   |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         | | |     | |   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue OMPCapturedExpr 0x{{.+}} '.capture_expr.' 'int'
// CHECK:         | | |     | |   |   `-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK:         | | |     | |   |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '+'
// CHECK:         | | |     | |   |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK:         | | |     | |   |       | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         | | |     | |   |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK:         | | |     | |   |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK:         | | |     | |   `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK:         | | |     | `-ImplicitCastExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'long' <IntegralCast>
// CHECK:         | | |     |   `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '/'
// CHECK:         | | |     |     |-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK:         | | |     |     | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK:         | | |     |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         | | |     |     |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue OMPCapturedExpr 0x{{.+}} '.capture_expr.' 'int'
// CHECK:         | | |     |     |   `-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK:         | | |     |     |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '+'
// CHECK:         | | |     |     |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK:         | | |     |     |       | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         | | |     |     |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK:         | | |     |     |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK:         | | |     |     `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK:         | | |     `-ImplicitCastExpr 0x{{.+}} <<invalid sloc>> 'long' <IntegralCast>
// CHECK:         | | |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK:         | | |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | | `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         | |-attrDetails: AlwaysInlineAttr 0x{{.+}} <<invalid sloc>> Implicit __forceinline
// CHECK:         | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int'
// CHECK:         | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .part_id. 'const int *const restrict'
// CHECK:         | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK:         | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:         | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .privates. 'void *const restrict'
// CHECK:         | | |-qualTypeDetail: QualType 0x{{.+}} 'void *const restrict' const __restrict
// CHECK:         | | | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK:         | | |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK:         | | `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK:         | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .copy_fn. 'void (*const restrict)(void *const restrict, ...)'
// CHECK:         | | |-qualTypeDetail: QualType 0x{{.+}} 'void (*const restrict)(void *const restrict, ...)' const __restrict
// CHECK:         | | | `-typeDetails: PointerType 0x{{.+}} 'void (*)(void *const restrict, ...)'
// CHECK:         | | |   `-typeDetails: FunctionProtoType 0x{{.+}} 'void (void *const restrict, ...)' variadic cdecl
// CHECK:         | | |     |-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK:         | | |     |-functionDetails:  cdeclReturnType 0x{{.+}} 'void'
// CHECK:         | | |     |-qualTypeDetail: QualType 0x{{.+}} 'void *const restrict' const __restrict
// CHECK:         | | |     | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK:         | | |     |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK:         | | |     `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK:         | | |-QualType 0x{{.+}} 'void (void *const restrict, ...)'
// CHECK:         | | `-typeDetails: FunctionProtoType 0x{{.+}} 'void (void *const restrict, ...)' variadic cdecl
// CHECK:         | |   |-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK:         | |   |-functionDetails:  cdeclReturnType 0x{{.+}} 'void'
// CHECK:         | |   |-qualTypeDetail: QualType 0x{{.+}} 'void *const restrict' const __restrict
// CHECK:         | |   | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK:         | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK:         | |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK:         | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .task_t. 'void *const'
// CHECK:         | | |-qualTypeDetail: QualType 0x{{.+}} 'void *const' const
// CHECK:         | | | `-typeDetails: PointerType 0x{{.+}} 'void *'
// CHECK:         | | |   `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK:         | | `-typeDetails: BuiltinType 0x{{.+}} 'void'
// CHECK:         | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK:         | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK:         | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK:         | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK:         | | |     `-Record 0x{{.+}} ''
// CHECK:         | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK:         | |   `-Record 0x{{.+}} ''
// CHECK:         | |-RecordDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit struct definition
// CHECK:         | | |-attrDetails: CapturedRecordAttr 0x{{.+}} <<invalid sloc>> Implicit
// CHECK:         | | |-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int'
// CHECK:         | | | `-attrDetails: OMPCaptureKindAttr 0x{{.+}} <<invalid sloc>> Implicit 36
// CHECK:         | | |-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int'
// CHECK:         | | | `-attrDetails: OMPCaptureKindAttr 0x{{.+}} <<invalid sloc>> Implicit 36
// CHECK:         | | `-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int'
// CHECK:         | |   `-attrDetails: OMPCaptureKindAttr 0x{{.+}} <<invalid sloc>> Implicit 36
// CHECK:         | `-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         |   |-OMPTeamsDistributeDirective 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:         |   | |-OMPCollapseClause 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:         |   | | `-ConstantExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK:         |   | |   |-value: Int 2
// CHECK:         |   | |   `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 2
// CHECK:         |   | `-CapturedStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:         |   |   |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         |   |   | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:         |   |   | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:         |   |   | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         |   |   | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         |   |   | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   |   | | |-<<<NULL>>>
// CHECK:         |   |   | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK:         |   |   | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |   |   | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         |   |   | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |   |   | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK:         |   |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         |   |   | | `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:         |   |   | |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:         |   |   | |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         |   |   | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         |   |   | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   |   | |   |-<<<NULL>>>
// CHECK:         |   |   | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK:         |   |   | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |   |   | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         |   |   | |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |   |   | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   | |   |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK:         |   |   | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         |   |   | |   `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:         |   |   | |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:         |   |   | |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         |   |   | |     |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         |   |   | |     |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   |   | |     |-<<<NULL>>>
// CHECK:         |   |   | |     |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK:         |   |   | |     | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |   |   | |     | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         |   |   | |     | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |   |   | |     |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   | |     |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK:         |   |   | |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         |   |   | |     `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:         |   |   | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK:         |   |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK:         |   |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:         |   |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         |   |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         |   |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK:         |   |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK:         |   |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:         |   |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         |   |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         |   |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK:         |   |   | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK:         |   |   | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK:         |   |   | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK:         |   |   | | |     `-Record 0x{{.+}} ''
// CHECK:         |   |   | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK:         |   |   | |   `-Record 0x{{.+}} ''
// CHECK:         |   |   | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         |   |   | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         |   |   | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   |   | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         |   |   | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         |   |   | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   |   | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         |   |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         |   |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   |   |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |   `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} 
// CHECK:         |   | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK:         |   | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK:         |   | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK:         |   | |     `-Record 0x{{.+}} ''
// CHECK:         |   | `-typeDetails: RecordType 0x{{.+}} 
// CHECK:         |   |   `-Record 0x{{.+}} ''
// CHECK:         |   |-RecordDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit struct definition
// CHECK:         |   | |-attrDetails: CapturedRecordAttr 0x{{.+}} <<invalid sloc>> Implicit
// CHECK:         |   | |-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK:         |   | |-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK:         |   | `-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit 'int &'
// CHECK:         |   |-CapturedDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK:         |   | |-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:         |   | | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:         |   | | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         |   | | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         |   | | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   | | |-<<<NULL>>>
// CHECK:         |   | | |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK:         |   | | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |   | | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         |   | | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |   | | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   | | |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK:         |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         |   | | `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:         |   | |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:         |   | |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         |   | |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         |   | |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   | |   |-<<<NULL>>>
// CHECK:         |   | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK:         |   | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |   | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         |   | |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |   | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   | |   |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK:         |   | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         |   | |   `-ForStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:         |   | |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:         |   | |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         |   | |     |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         |   | |     |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   | |     |-<<<NULL>>>
// CHECK:         |   | |     |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '<'
// CHECK:         |   | |     | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |   | |     | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         |   | |     | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |   | |     |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   | |     |-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' postfix '++'
// CHECK:         |   | |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'i' 'int'
// CHECK:         |   | |     `-NullStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:         |   | |-ImplicitParamDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK:         |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK:         |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:         |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK:         |   | | |-qualTypeDetail: QualType 0x{{.+}} 'const int *const restrict' const __restrict
// CHECK:         |   | | | `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:         |   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         |   | | |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   | | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         |   | |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   | |-ImplicitParamDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 
// CHECK:         |   | | |-qualTypeDetail: QualType 0x{{.+}} 
// CHECK:         |   | | | `-typeDetails: PointerType 0x{{.+}} 
// CHECK:         |   | | |   `-typeDetails: RecordType 0x{{.+}} 
// CHECK:         |   | | |     `-Record 0x{{.+}} ''
// CHECK:         |   | | `-typeDetails: RecordType 0x{{.+}} 
// CHECK:         |   | |   `-Record 0x{{.+}} ''
// CHECK:         |   | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         |   | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         |   | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   | |-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         |   | | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         |   | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   | `-VarDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used i 'int' cinit
// CHECK:         |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         |   |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |   |-OMPCapturedExprDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK:         |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   |-OMPCapturedExprDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit used .capture_expr. 'int'
// CHECK:         |   | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK:         |   `-OMPCapturedExprDecl 0x{{.+}} <line:{{.*}}:{{.*}}, <invalid sloc>> col:{{.*}} implicit used .capture_expr. 'long'
// CHECK:         |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'long' '-'
// CHECK:         |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}> 'long' '*'
// CHECK:         |       | |-ImplicitCastExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'long' <IntegralCast>
// CHECK:         |       | | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '/'
// CHECK:         |       | |   |-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK:         |       | |   | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK:         |       | |   |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |       | |   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue OMPCapturedExpr 0x{{.+}} '.capture_expr.' 'int'
// CHECK:         |       | |   |   `-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK:         |       | |   |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '+'
// CHECK:         |       | |   |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK:         |       | |   |       | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         |       | |   |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK:         |       | |   |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK:         |       | |   `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK:         |       | `-ImplicitCastExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'long' <IntegralCast>
// CHECK:         |       |   `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '/'
// CHECK:         |       |     |-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK:         |       |     | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK:         |       |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |       |     |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue OMPCapturedExpr 0x{{.+}} '.capture_expr.' 'int'
// CHECK:         |       |     |   `-ParenExpr 0x{{.+}} <col:{{.*}}> 'int'
// CHECK:         |       |     |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, <invalid sloc>> 'int' '+'
// CHECK:         |       |     |       |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int' '-'
// CHECK:         |       |     |       | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:         |       |     |       | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK:         |       |     |       `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK:         |       |     `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK:         |       `-ImplicitCastExpr 0x{{.+}} <<invalid sloc>> 'long' <IntegralCast>
// CHECK:         |         `-IntegerLiteral 0x{{.+}} <<invalid sloc>> 'int' 1
// CHECK:         |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'x' 'int'
// CHECK:         |-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'y' 'int'
// CHECK:         `-DeclRefExpr 0x{{.+}} <line:{{.*}}:{{.*}}> 'int' lvalue ParmVar 0x{{.+}} 'z' 'int'