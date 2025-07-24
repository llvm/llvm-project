// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -ast-dump %s | FileCheck --match-full-lines -implicit-check-not=openmp_structured_block %s

void test_one(int x) {
#pragma omp distribute parallel for
  for (int i = 0; i < x; i++)
    ;
}

void test_two(int x, int y) {
#pragma omp distribute parallel for
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_three(int x, int y) {
#pragma omp distribute parallel for collapse(1)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_four(int x, int y) {
#pragma omp distribute parallel for collapse(2)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_five(int x, int y, int z) {
#pragma omp distribute parallel for collapse(2)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      for (int i = 0; i < z; i++)
        ;
}

// CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NEXT: |-TypedefDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __int128_t '__int128'
// CHECK-NEXT: | `-typeDetails: BuiltinType {{.*}} '__int128'
// CHECK-NEXT: |-TypedefDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __uint128_t 'unsigned __int128'
// CHECK-NEXT: | `-typeDetails: BuiltinType {{.*}} 'unsigned __int128'
// CHECK-NEXT: |-TypedefDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __NSConstantString 'struct __NSConstantString_tag'
// CHECK-NEXT: | `-typeDetails: RecordType {{.*}} 'struct __NSConstantString_tag'
// CHECK-NEXT: |   `-Record {{.*}} '__NSConstantString_tag'
// CHECK-NEXT: |-TypedefDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __builtin_ms_va_list 'char *'
// CHECK-NEXT: | `-typeDetails: PointerType {{.*}} 'char *'
// CHECK-NEXT: |   `-typeDetails: BuiltinType {{.*}} 'char'
// CHECK-NEXT: |-TypedefDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __builtin_va_list 'struct __va_list_tag[1]'
// CHECK-NEXT: | `-typeDetails: ConstantArrayType {{.*}} 'struct __va_list_tag[1]' 1 
// CHECK-NEXT: |   `-typeDetails: RecordType {{.*}} 'struct __va_list_tag'
// CHECK-NEXT: |     `-Record {{.*}} '__va_list_tag'
// CHECK-NEXT: |-FunctionDecl {{.*}} test_one 'void (int)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used x 'int'
// CHECK-NEXT: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-OMPDistributeParallelForDirective {{.*}} 
// CHECK-NEXT: |     `-CapturedStmt {{.*}} 
// CHECK-NEXT: |       |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-ForStmt {{.*}} 
// CHECK-NEXT: |       | | |-DeclStmt {{.*}} 
// CHECK-NEXT: |       | | | `-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT: |       | | |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |       | | |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |-BinaryOperator {{.*}} 'int' '<'
// CHECK-NEXT: |       | | | |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: |       | | |-UnaryOperator {{.*}} 'int' postfix '++'
// CHECK-NEXT: |       | | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | `-NullStmt {{.*}} 
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |-qualTypeDetail: QualType {{.*}} 'const int *const restrict' const __restrict
// CHECK-NEXT: |       | | | `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT: |       | | |   `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |       | | |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | | `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |       | |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |-qualTypeDetail: QualType {{.*}} 'const int *const restrict' const __restrict
// CHECK-NEXT: |       | | | `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT: |       | | |   `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |       | | |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | | `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |       | |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit used .previous.lb. 'const __size_t':'const unsigned long'
// CHECK-NEXT: |       | | `-qualTypeDetail: QualType {{.*}} 'const __size_t' const
// CHECK-NEXT: |       | |   `-typeDetails: PredefinedSugarType {{.*}} '__size_t' sugar
// CHECK-NEXT: |       | |     `-typeDetails: BuiltinType {{.*}} 'unsigned long'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit used .previous.ub. 'const __size_t':'const unsigned long'
// CHECK-NEXT: |       | | `-qualTypeDetail: QualType {{.*}} 'const __size_t' const
// CHECK-NEXT: |       | |   `-typeDetails: PredefinedSugarType {{.*}} '__size_t' sugar
// CHECK-NEXT: |       | |     `-typeDetails: BuiltinType {{.*}} 'unsigned long'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit __context 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:4:1) *const restrict'
// CHECK-NEXT: |       | | |-qualTypeDetail: QualType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:4:1) *const restrict' const __restrict
// CHECK-NEXT: |       | | | `-typeDetails: PointerType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:4:1) *'
// CHECK-NEXT: |       | | |   `-typeDetails: RecordType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:4:1)'
// CHECK-NEXT: |       | | |     `-Record {{.*}} 
// CHECK-NEXT: |       | | `-typeDetails: RecordType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:4:1)'
// CHECK-NEXT: |       | |   `-Record {{.*}} 
// CHECK-NEXT: |       | `-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT: |       |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |       |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |-FunctionDecl {{.*}} test_two 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used x 'int'
// CHECK-NEXT: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used y 'int'
// CHECK-NEXT: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-OMPDistributeParallelForDirective {{.*}} 
// CHECK-NEXT: |     `-CapturedStmt {{.*}} 
// CHECK-NEXT: |       |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-ForStmt {{.*}} 
// CHECK-NEXT: |       | | |-DeclStmt {{.*}} 
// CHECK-NEXT: |       | | | `-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT: |       | | |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |       | | |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |-BinaryOperator {{.*}} 'int' '<'
// CHECK-NEXT: |       | | | |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: |       | | |-UnaryOperator {{.*}} 'int' postfix '++'
// CHECK-NEXT: |       | | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | `-ForStmt {{.*}} 
// CHECK-NEXT: |       | |   |-DeclStmt {{.*}} 
// CHECK-NEXT: |       | |   | `-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT: |       | |   |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |       | |   |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | |   |-<<<NULL>>>
// CHECK-NEXT: |       | |   |-BinaryOperator {{.*}} 'int' '<'
// CHECK-NEXT: |       | |   | |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | |   | `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   |   `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: |       | |   |-UnaryOperator {{.*}} 'int' postfix '++'
// CHECK-NEXT: |       | |   | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | |   `-NullStmt {{.*}} 
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |-qualTypeDetail: QualType {{.*}} 'const int *const restrict' const __restrict
// CHECK-NEXT: |       | | | `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT: |       | | |   `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |       | | |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | | `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |       | |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |-qualTypeDetail: QualType {{.*}} 'const int *const restrict' const __restrict
// CHECK-NEXT: |       | | | `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT: |       | | |   `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |       | | |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | | `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |       | |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit used .previous.lb. 'const __size_t':'const unsigned long'
// CHECK-NEXT: |       | | `-qualTypeDetail: QualType {{.*}} 'const __size_t' const
// CHECK-NEXT: |       | |   `-typeDetails: PredefinedSugarType {{.*}} '__size_t' sugar
// CHECK-NEXT: |       | |     `-typeDetails: BuiltinType {{.*}} 'unsigned long'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit used .previous.ub. 'const __size_t':'const unsigned long'
// CHECK-NEXT: |       | | `-qualTypeDetail: QualType {{.*}} 'const __size_t' const
// CHECK-NEXT: |       | |   `-typeDetails: PredefinedSugarType {{.*}} '__size_t' sugar
// CHECK-NEXT: |       | |     `-typeDetails: BuiltinType {{.*}} 'unsigned long'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit __context 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:10:1) *const restrict'
// CHECK-NEXT: |       | | |-qualTypeDetail: QualType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:10:1) *const restrict' const __restrict
// CHECK-NEXT: |       | | | `-typeDetails: PointerType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:10:1) *'
// CHECK-NEXT: |       | | |   `-typeDetails: RecordType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:10:1)'
// CHECK-NEXT: |       | | |     `-Record {{.*}} 
// CHECK-NEXT: |       | | `-typeDetails: RecordType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:10:1)'
// CHECK-NEXT: |       | |   `-Record {{.*}} 
// CHECK-NEXT: |       | |-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT: |       | | |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |       | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | `-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT: |       |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |       |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       |-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |-FunctionDecl {{.*}} test_three 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used x 'int'
// CHECK-NEXT: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used y 'int'
// CHECK-NEXT: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-OMPDistributeParallelForDirective {{.*}} 
// CHECK-NEXT: |     |-OMPCollapseClause {{.*}} 
// CHECK-NEXT: |     | `-ConstantExpr {{.*}} 'int'
// CHECK-NEXT: |     |   |-value: Int 1
// CHECK-NEXT: |     |   `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: |     `-CapturedStmt {{.*}} 
// CHECK-NEXT: |       |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-ForStmt {{.*}} 
// CHECK-NEXT: |       | | |-DeclStmt {{.*}} 
// CHECK-NEXT: |       | | | `-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT: |       | | |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |       | | |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |-BinaryOperator {{.*}} 'int' '<'
// CHECK-NEXT: |       | | | |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: |       | | |-UnaryOperator {{.*}} 'int' postfix '++'
// CHECK-NEXT: |       | | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | `-ForStmt {{.*}} 
// CHECK-NEXT: |       | |   |-DeclStmt {{.*}} 
// CHECK-NEXT: |       | |   | `-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT: |       | |   |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |       | |   |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | |   |-<<<NULL>>>
// CHECK-NEXT: |       | |   |-BinaryOperator {{.*}} 'int' '<'
// CHECK-NEXT: |       | |   | |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | |   | `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   |   `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: |       | |   |-UnaryOperator {{.*}} 'int' postfix '++'
// CHECK-NEXT: |       | |   | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | |   `-NullStmt {{.*}} 
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |-qualTypeDetail: QualType {{.*}} 'const int *const restrict' const __restrict
// CHECK-NEXT: |       | | | `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT: |       | | |   `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |       | | |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | | `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |       | |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |-qualTypeDetail: QualType {{.*}} 'const int *const restrict' const __restrict
// CHECK-NEXT: |       | | | `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT: |       | | |   `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |       | | |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | | `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |       | |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit used .previous.lb. 'const __size_t':'const unsigned long'
// CHECK-NEXT: |       | | `-qualTypeDetail: QualType {{.*}} 'const __size_t' const
// CHECK-NEXT: |       | |   `-typeDetails: PredefinedSugarType {{.*}} '__size_t' sugar
// CHECK-NEXT: |       | |     `-typeDetails: BuiltinType {{.*}} 'unsigned long'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit used .previous.ub. 'const __size_t':'const unsigned long'
// CHECK-NEXT: |       | | `-qualTypeDetail: QualType {{.*}} 'const __size_t' const
// CHECK-NEXT: |       | |   `-typeDetails: PredefinedSugarType {{.*}} '__size_t' sugar
// CHECK-NEXT: |       | |     `-typeDetails: BuiltinType {{.*}} 'unsigned long'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit __context 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:17:1) *const restrict'
// CHECK-NEXT: |       | | |-qualTypeDetail: QualType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:17:1) *const restrict' const __restrict
// CHECK-NEXT: |       | | | `-typeDetails: PointerType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:17:1) *'
// CHECK-NEXT: |       | | |   `-typeDetails: RecordType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:17:1)'
// CHECK-NEXT: |       | | |     `-Record {{.*}} 
// CHECK-NEXT: |       | | `-typeDetails: RecordType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:17:1)'
// CHECK-NEXT: |       | |   `-Record {{.*}} 
// CHECK-NEXT: |       | |-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT: |       | | |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |       | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | `-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT: |       |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |       |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       |-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: |-FunctionDecl {{.*}} test_four 'void (int, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used x 'int'
// CHECK-NEXT: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} used y 'int'
// CHECK-NEXT: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-OMPDistributeParallelForDirective {{.*}} 
// CHECK-NEXT: |     |-OMPCollapseClause {{.*}} 
// CHECK-NEXT: |     | `-ConstantExpr {{.*}} 'int'
// CHECK-NEXT: |     |   |-value: Int 2
// CHECK-NEXT: |     |   `-IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT: |     `-CapturedStmt {{.*}} 
// CHECK-NEXT: |       |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT: |       | |-ForStmt {{.*}} 
// CHECK-NEXT: |       | | |-DeclStmt {{.*}} 
// CHECK-NEXT: |       | | | `-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT: |       | | |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |       | | |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | | |-<<<NULL>>>
// CHECK-NEXT: |       | | |-BinaryOperator {{.*}} 'int' '<'
// CHECK-NEXT: |       | | | |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |       | | | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | | `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |       | | |   `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: |       | | |-UnaryOperator {{.*}} 'int' postfix '++'
// CHECK-NEXT: |       | | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | | `-ForStmt {{.*}} 
// CHECK-NEXT: |       | |   |-DeclStmt {{.*}} 
// CHECK-NEXT: |       | |   | `-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT: |       | |   |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |       | |   |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | |   |-<<<NULL>>>
// CHECK-NEXT: |       | |   |-BinaryOperator {{.*}} 'int' '<'
// CHECK-NEXT: |       | |   | |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | |   | `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |       | |   |   `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT: |       | |   |-UnaryOperator {{.*}} 'int' postfix '++'
// CHECK-NEXT: |       | |   | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: |       | |   `-NullStmt {{.*}} 
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |-qualTypeDetail: QualType {{.*}} 'const int *const restrict' const __restrict
// CHECK-NEXT: |       | | | `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT: |       | | |   `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |       | | |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | | `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |       | |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT: |       | | |-qualTypeDetail: QualType {{.*}} 'const int *const restrict' const __restrict
// CHECK-NEXT: |       | | | `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT: |       | | |   `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |       | | |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | | `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |       | |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit used .previous.lb. 'const __size_t':'const unsigned long'
// CHECK-NEXT: |       | | `-qualTypeDetail: QualType {{.*}} 'const __size_t' const
// CHECK-NEXT: |       | |   `-typeDetails: PredefinedSugarType {{.*}} '__size_t' sugar
// CHECK-NEXT: |       | |     `-typeDetails: BuiltinType {{.*}} 'unsigned long'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit used .previous.ub. 'const __size_t':'const unsigned long'
// CHECK-NEXT: |       | | `-qualTypeDetail: QualType {{.*}} 'const __size_t' const
// CHECK-NEXT: |       | |   `-typeDetails: PredefinedSugarType {{.*}} '__size_t' sugar
// CHECK-NEXT: |       | |     `-typeDetails: BuiltinType {{.*}} 'unsigned long'
// CHECK-NEXT: |       | |-ImplicitParamDecl {{.*}} implicit __context 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:24:1) *const restrict'
// CHECK-NEXT: |       | | |-qualTypeDetail: QualType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:24:1) *const restrict' const __restrict
// CHECK-NEXT: |       | | | `-typeDetails: PointerType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:24:1) *'
// CHECK-NEXT: |       | | |   `-typeDetails: RecordType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:24:1)'
// CHECK-NEXT: |       | | |     `-Record {{.*}} 
// CHECK-NEXT: |       | | `-typeDetails: RecordType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:24:1)'
// CHECK-NEXT: |       | |   `-Record {{.*}} 
// CHECK-NEXT: |       | |-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT: |       | | |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |       | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       | `-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT: |       |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |       |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |       |-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT: |       `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT: `-FunctionDecl {{.*}} test_five 'void (int, int, int)'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} used x 'int'
// CHECK-NEXT:   | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} used y 'int'
// CHECK-NEXT:   | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:   |-ParmVarDecl {{.*}} used z 'int'
// CHECK-NEXT:   | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:   `-CompoundStmt {{.*}} 
// CHECK-NEXT:     `-OMPDistributeParallelForDirective {{.*}} 
// CHECK-NEXT:       |-OMPCollapseClause {{.*}} 
// CHECK-NEXT:       | `-ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       |   |-value: Int 2
// CHECK-NEXT:       |   `-IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT:       `-CapturedStmt {{.*}} 
// CHECK-NEXT:         |-CapturedDecl {{.*}} <<invalid sloc>> <invalid sloc> nothrow
// CHECK-NEXT:         | |-ForStmt {{.*}} 
// CHECK-NEXT:         | | |-DeclStmt {{.*}} 
// CHECK-NEXT:         | | | `-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT:         | | |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT:         | | |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:         | | |-<<<NULL>>>
// CHECK-NEXT:         | | |-BinaryOperator {{.*}} 'int' '<'
// CHECK-NEXT:         | | | |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:         | | | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | | `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:         | | |   `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT:         | | |-UnaryOperator {{.*}} 'int' postfix '++'
// CHECK-NEXT:         | | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | | `-ForStmt {{.*}} 
// CHECK-NEXT:         | |   |-DeclStmt {{.*}} 
// CHECK-NEXT:         | |   | `-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT:         | |   |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT:         | |   |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:         | |   |-<<<NULL>>>
// CHECK-NEXT:         | |   |-BinaryOperator {{.*}} 'int' '<'
// CHECK-NEXT:         | |   | |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:         | |   | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | |   | `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:         | |   |   `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'y' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT:         | |   |-UnaryOperator {{.*}} 'int' postfix '++'
// CHECK-NEXT:         | |   | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | |   `-ForStmt {{.*}} 
// CHECK-NEXT:         | |     |-DeclStmt {{.*}} 
// CHECK-NEXT:         | |     | `-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT:         | |     |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT:         | |     |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:         | |     |-<<<NULL>>>
// CHECK-NEXT:         | |     |-BinaryOperator {{.*}} 'int' '<'
// CHECK-NEXT:         | |     | |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:         | |     | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | |     | `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:         | |     |   `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'z' 'int' refers_to_enclosing_variable_or_capture
// CHECK-NEXT:         | |     |-UnaryOperator {{.*}} 'int' postfix '++'
// CHECK-NEXT:         | |     | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:         | |     `-NullStmt {{.*}} 
// CHECK-NEXT:         | |-ImplicitParamDecl {{.*}} implicit .global_tid. 'const int *const restrict'
// CHECK-NEXT:         | | |-qualTypeDetail: QualType {{.*}} 'const int *const restrict' const __restrict
// CHECK-NEXT:         | | | `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT:         | | |   `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT:         | | |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:         | | `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT:         | |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:         | |-ImplicitParamDecl {{.*}} implicit .bound_tid. 'const int *const restrict'
// CHECK-NEXT:         | | |-qualTypeDetail: QualType {{.*}} 'const int *const restrict' const __restrict
// CHECK-NEXT:         | | | `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT:         | | |   `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT:         | | |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:         | | `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT:         | |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:         | |-ImplicitParamDecl {{.*}} implicit used .previous.lb. 'const __size_t':'const unsigned long'
// CHECK-NEXT:         | | `-qualTypeDetail: QualType {{.*}} 'const __size_t' const
// CHECK-NEXT:         | |   `-typeDetails: PredefinedSugarType {{.*}} '__size_t' sugar
// CHECK-NEXT:         | |     `-typeDetails: BuiltinType {{.*}} 'unsigned long'
// CHECK-NEXT:         | |-ImplicitParamDecl {{.*}} implicit used .previous.ub. 'const __size_t':'const unsigned long'
// CHECK-NEXT:         | | `-qualTypeDetail: QualType {{.*}} 'const __size_t' const
// CHECK-NEXT:         | |   `-typeDetails: PredefinedSugarType {{.*}} '__size_t' sugar
// CHECK-NEXT:         | |     `-typeDetails: BuiltinType {{.*}} 'unsigned long'
// CHECK-NEXT:         | |-ImplicitParamDecl {{.*}} implicit __context 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:31:1) *const restrict'
// CHECK-NEXT:         | | |-qualTypeDetail: QualType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:31:1) *const restrict' const __restrict
// CHECK-NEXT:         | | | `-typeDetails: PointerType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:31:1) *'
// CHECK-NEXT:         | | |   `-typeDetails: RecordType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:31:1)'
// CHECK-NEXT:         | | |     `-Record {{.*}} 
// CHECK-NEXT:         | | `-typeDetails: RecordType {{.*}} 'struct (unnamed at C:\Precision\sei\llvm-patch\llvm-project\clang\test\AST\ast-dump-openmp-distribute-parallel-for.c:31:1)'
// CHECK-NEXT:         | |   `-Record {{.*}} 
// CHECK-NEXT:         | |-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT:         | | |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT:         | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:         | |-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT:         | | |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT:         | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:         | `-VarDecl {{.*}} used i 'int' cinit
// CHECK-NEXT:         |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT:         |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:         |-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK-NEXT:         |-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'y' 'int'
// CHECK-NEXT:         `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'z' 'int'
