// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fenable-ripple -ast-dump %s | FileCheck --match-full-lines %s

#define ripple_set_block_shape(PEId, Size) \
  (ripple_block_t) __builtin_ripple_set_shape((PEId), (Size), 1, 1, 1, 1, 1, 1, 1, 1, 1)

typedef struct ripple_block_shape *ripple_block_t;

void test_one(int x) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
#pragma ripple parallel Block(BS) Dims(0)
  for (int i = 0; i < x; i++)
    ;
}

void test_two(int x, int y) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
#pragma ripple parallel Block(BS) Dims(0)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      ;
}

void test_three(int x, int y) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
#pragma ripple parallel Block(BS) Dims(0)
  for (int i = 0; i < x; i++)
#pragma ripple parallel Block(BS) Dims(1)
    for (int i = 0; i < y; i++)
      ;
}

void test_four(int x, int y) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
  for (int i = 0; i < x; i++)
#pragma ripple parallel Block(BS) Dims(0)
    for (int i = 0; i < y; i++)
      ;
}

void test_five(int x, int y, int z) {
  ripple_block_t BS = ripple_set_block_shape(0, 4);
#pragma ripple parallel Block(BS) Dims(0, 1)
  for (int i = 0; i < x; i++)
    for (int i = 0; i < y; i++)
      for (int i = 0; i < z; i++)
        ;
}

//CHECK: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
//CHECK: |-RecordDecl {{.*}} <{{.*}}ast-dump-ripple-parallel.c:6:9, col:16> col:16 struct ripple_block_shape
//CHECK-NEXT: |-TypedefDecl {{.*}} <col:1, col:36> col:36 referenced ripple_block_t 'struct ripple_block_shape *'
//CHECK-NEXT: | `-PointerType {{.*}} 'struct ripple_block_shape *'
//CHECK-NEXT: |     `-RecordType {{.*}} 'struct ripple_block_shape' owns_tag struct
//CHECK-NEXT: |       `-Record {{.*}} 'ripple_block_shape'
//CHECK-NEXT: |-FunctionDecl {{.*}} <line:8:1, line:13:1> line:8:6 test_one 'void (int)'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <col:15, col:19> col:19 used x 'int'
//CHECK-NEXT: | `-CompoundStmt {{.*}} <col:22, line:13:1>
//CHECK-NEXT: |   |-DeclStmt {{.*}} <line:9:3, col:51>
//CHECK-NEXT: |   | `-VarDecl {{.*}} <col:3, line:4:88> line:9:18 used BS 'ripple_block_t':'struct ripple_block_shape *' cinit
//CHECK-NEXT: |   |   `-CStyleCastExpr {{.*}} <line:4:3, col:88> 'ripple_block_t':'struct ripple_block_shape *' <BitCast>
//CHECK-NEXT: |   |     `-CallExpr {{.*}} <col:20, col:88> 'void *'
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:20> 'void *(*)(__size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t)' <BuiltinFnToFnPtr>
//CHECK-NEXT: |   |       | `-DeclRefExpr {{.*}} <col:20> '<builtin fn type>' Function {{.*}} '__builtin_ripple_set_shape' 'void *(__size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t)'
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:47, col:52> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-ParenExpr {{.*}} <col:47, col:52> 'int'
//CHECK-NEXT: |   |       |   `-IntegerLiteral {{.*}} <line:9:46> 'int' 0
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <line:4:55, col:60> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-ParenExpr {{.*}} <col:55, col:60> 'int'
//CHECK-NEXT: |   |       |   `-IntegerLiteral {{.*}} <line:9:49> 'int' 4
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <line:4:63> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:63> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:66> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:66> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:69> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:69> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:72> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:72> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:75> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:75> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:78> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:78> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:81> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:81> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:84> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:84> 'int' 1
//CHECK-NEXT: |   |       `-ImplicitCastExpr {{.*}} <col:87> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |         `-IntegerLiteral {{.*}} <col:87> 'int' 1
//CHECK-NEXT: |   `-RippleComputeConstruct {{.*}} <line:10:1, col:42>
//CHECK-NEXT: |     |-ForStmt {{.*}} <line:11:3, line:12:5>
//CHECK-NEXT: |     | |-DeclStmt {{.*}} <line:11:8, col:17>
//CHECK-NEXT: |     | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
//CHECK-NEXT: |     | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
//CHECK-NEXT: |     | |-<<<NULL>>>
//CHECK-NEXT: |     | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
//CHECK-NEXT: |     | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
//CHECK-NEXT: |     | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
//CHECK-NEXT: |     | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
//CHECK-NEXT: |     | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
//CHECK-NEXT: |     | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     | `-NullStmt {{.*}} <line:12:5>
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <line:10:35> 'int' lvalue Var {{.*}} 'ripple.par.block.size' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <line:11:8> 'int' lvalue Var {{.*}} 'ripple.loop.iters' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.loop.iters' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |-BinaryOperator {{.*}} <col:8, col:17> 'int' '='
//CHECK-NEXT: |     | |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     | `-BinaryOperator {{.*}} <col:8, col:17> 'int' '+'
//CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   | `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |     |   `-ParenExpr {{.*}} <col:8, col:17> 'int'
//CHECK-NEXT: |     |     `-BinaryOperator {{.*}} <col:8, col:3> 'int' '*'
//CHECK-NEXT: |     |       |-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     |       | `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |     |       `-ImplicitCastExpr {{.*}} <col:3> 'int' <LValueToRValue>
//CHECK-NEXT: |     |         `-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |-ForStmt {{.*}} <col:3, line:12:5>
//CHECK-NEXT: |     | |-<<<NULL>>>
//CHECK-NEXT: |     | |-<<<NULL>>>
//CHECK-NEXT: |     | |-BinaryOperator {{.*}} <line:11:3, col:8> 'int' '<'
//CHECK-NEXT: |     | | |-ImplicitCastExpr {{.*}} <col:3> 'int' <LValueToRValue>
//CHECK-NEXT: |     | | | `-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     | | `-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     | |   `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.loop.iters' 'int'
//CHECK-NEXT: |     | |-CompoundAssignOperator {{.*}} <col:3, col:26> 'int' '+=' ComputeLHSTy='int' ComputeResultTy='int'
//CHECK-NEXT: |     | | |-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     | | `-IntegerLiteral {{.*}} <col:26> 'int' 1
//CHECK-NEXT: |     | `-CompoundStmt {{.*}} <line:12:5>
//CHECK-NEXT: |     |   |-BinaryOperator {{.*}} <line:11:8, col:17> 'int' '='
//CHECK-NEXT: |     |   | |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |   | `-BinaryOperator {{.*}} <col:8, col:17> 'int' '+'
//CHECK-NEXT: |     |   |   |-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |   | `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |     |   |   `-ParenExpr {{.*}} <col:8, col:17> 'int'
//CHECK-NEXT: |     |   |     `-BinaryOperator {{.*}} <col:8, col:3> 'int' '*'
//CHECK-NEXT: |     |   |       |-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |       | `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |     |   |       `-ImplicitCastExpr {{.*}} <col:3> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |         `-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |   `-NullStmt {{.*}} <line:12:5>
//CHECK-NEXT: |     |-BinaryOperator {{.*}} <line:11:3, col:8> 'int' '!='
//CHECK-NEXT: |     | |-BinaryOperator {{.*}} <col:3, line:10:35> 'int' '*'
//CHECK-NEXT: |     | | |-ImplicitCastExpr {{.*}} <line:11:3> 'int' <LValueToRValue>
//CHECK-NEXT: |     | | | `-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     | | `-ImplicitCastExpr {{.*}} <line:10:35> 'int' <LValueToRValue>
//CHECK-NEXT: |     | |   `-DeclRefExpr {{.*}} <col:35> 'int' lvalue Var {{.*}} 'ripple.par.block.size' 'int'
//CHECK-NEXT: |     | `-ImplicitCastExpr {{.*}} <line:11:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.loop.iters' 'int'
//CHECK-NEXT: |     |-BinaryOperator {{.*}} <col:8, col:26> 'int' '='
//CHECK-NEXT: |     | |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     | `-BinaryOperator {{.*}} <col:16, col:26> 'int' '+'
//CHECK-NEXT: |     |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
//CHECK-NEXT: |     |   `-ParenExpr {{.*}} <col:26> 'int'
//CHECK-NEXT: |     |     `-BinaryOperator {{.*}} <col:3, col:26> 'int' '*'
//CHECK-NEXT: |     |       |-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
//CHECK-NEXT: |     |       | |-ParenExpr {{.*}} <col:3> 'int'
//CHECK-NEXT: |     |       | | `-BinaryOperator {{.*}} <col:23, col:3> 'int' '-'
//CHECK-NEXT: |     |       | |   |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
//CHECK-NEXT: |     |       | |   | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
//CHECK-NEXT: |     |       | |   `-ParenExpr {{.*}} <col:3> 'int'
//CHECK-NEXT: |     |       | |     `-BinaryOperator {{.*}} <col:16, <invalid sloc>> 'int' '+'
//CHECK-NEXT: |     |       | |       |-BinaryOperator {{.*}} <col:16, col:26> 'int' '-'
//CHECK-NEXT: |     |       | |       | |-IntegerLiteral {{.*}} <col:16> 'int' 0
//CHECK-NEXT: |     |       | |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
//CHECK-NEXT: |     |       | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
//CHECK-NEXT: |     |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
//CHECK-NEXT: |     |       `-IntegerLiteral {{.*}} <col:26> 'int' 1
//CHECK-NEXT: |     `-NullStmt {{.*}} <line:12:5>
//CHECK-NEXT: |-FunctionDecl {{.*}} <line:4:20> col:20 implicit used __builtin_ripple_set_shape 'void *(__size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t)' extern
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <<invalid sloc>> <invalid sloc> '__size_t':'unsigned long'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <<invalid sloc>> <invalid sloc> '__size_t':'unsigned long'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <<invalid sloc>> <invalid sloc> '__size_t':'unsigned long'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <<invalid sloc>> <invalid sloc> '__size_t':'unsigned long'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <<invalid sloc>> <invalid sloc> '__size_t':'unsigned long'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <<invalid sloc>> <invalid sloc> '__size_t':'unsigned long'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <<invalid sloc>> <invalid sloc> '__size_t':'unsigned long'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <<invalid sloc>> <invalid sloc> '__size_t':'unsigned long'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <<invalid sloc>> <invalid sloc> '__size_t':'unsigned long'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <<invalid sloc>> <invalid sloc> '__size_t':'unsigned long'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <<invalid sloc>> <invalid sloc> '__size_t':'unsigned long'
//CHECK-NEXT: | |-BuiltinAttr {{.*}} <<invalid sloc>> Implicit {{[0-9]+}}
//CHECK-NEXT: | `-NoThrowAttr {{.*}} <col:20> Implicit
//CHECK-NEXT: |-FunctionDecl {{.*}} <line:10:1> col:1 implicit used __builtin_ripple_get_index '__size_t (void *, __size_t)' extern
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <<invalid sloc>> <invalid sloc> 'void *'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <<invalid sloc>> <invalid sloc> '__size_t':'unsigned long'
//CHECK-NEXT: | |-BuiltinAttr {{.*}} <<invalid sloc>> Implicit {{[0-9]+}}
//CHECK-NEXT: | `-NoThrowAttr {{.*}} <col:1> Implicit
//CHECK-NEXT: |-FunctionDecl {{.*}} <col:1> col:1 implicit used __builtin_ripple_get_size '__size_t (void *, __size_t)' extern
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <<invalid sloc>> <invalid sloc> 'void *'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <<invalid sloc>> <invalid sloc> '__size_t':'unsigned long'
//CHECK-NEXT: | |-BuiltinAttr {{.*}} <<invalid sloc>> Implicit {{[0-9]+}}
//CHECK-NEXT: | `-NoThrowAttr {{.*}} <col:1> Implicit
//CHECK-NEXT: |-FunctionDecl {{.*}} <line:15:1, line:21:1> line:15:6 test_two 'void (int, int)'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <col:15, col:19> col:19 used x 'int'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <col:22, col:26> col:26 used y 'int'
//CHECK-NEXT: | `-CompoundStmt {{.*}} <col:29, line:21:1>
//CHECK-NEXT: |   |-DeclStmt {{.*}} <line:16:3, col:51>
//CHECK-NEXT: |   | `-VarDecl {{.*}} <col:3, line:4:88> line:16:18 used BS 'ripple_block_t':'struct ripple_block_shape *' cinit
//CHECK-NEXT: |   |   `-CStyleCastExpr {{.*}} <line:4:3, col:88> 'ripple_block_t':'struct ripple_block_shape *' <BitCast>
//CHECK-NEXT: |   |     `-CallExpr {{.*}} <col:20, col:88> 'void *'
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:20> 'void *(*)(__size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t)' <BuiltinFnToFnPtr>
//CHECK-NEXT: |   |       | `-DeclRefExpr {{.*}} <col:20> '<builtin fn type>' Function {{.*}} '__builtin_ripple_set_shape' 'void *(__size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t)'
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:47, col:52> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-ParenExpr {{.*}} <col:47, col:52> 'int'
//CHECK-NEXT: |   |       |   `-IntegerLiteral {{.*}} <line:16:46> 'int' 0
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <line:4:55, col:60> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-ParenExpr {{.*}} <col:55, col:60> 'int'
//CHECK-NEXT: |   |       |   `-IntegerLiteral {{.*}} <line:16:49> 'int' 4
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <line:4:63> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:63> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:66> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:66> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:69> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:69> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:72> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:72> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:75> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:75> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:78> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:78> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:81> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:81> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:84> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:84> 'int' 1
//CHECK-NEXT: |   |       `-ImplicitCastExpr {{.*}} <col:87> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |         `-IntegerLiteral {{.*}} <col:87> 'int' 1
//CHECK-NEXT: |   `-RippleComputeConstruct {{.*}} <line:17:1, col:42>
//CHECK-NEXT: |     |-ForStmt {{.*}} <line:18:3, line:20:7>
//CHECK-NEXT: |     | |-DeclStmt {{.*}} <line:18:8, col:17>
//CHECK-NEXT: |     | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
//CHECK-NEXT: |     | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
//CHECK-NEXT: |     | |-<<<NULL>>>
//CHECK-NEXT: |     | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
//CHECK-NEXT: |     | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
//CHECK-NEXT: |     | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
//CHECK-NEXT: |     | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
//CHECK-NEXT: |     | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
//CHECK-NEXT: |     | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     | `-ForStmt {{.*}} <line:19:5, line:20:7>
//CHECK-NEXT: |     |   |-DeclStmt {{.*}} <line:19:10, col:19>
//CHECK-NEXT: |     |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
//CHECK-NEXT: |     |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT: |     |   |-<<<NULL>>>
//CHECK-NEXT: |     |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
//CHECK-NEXT: |     |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
//CHECK-NEXT: |     |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
//CHECK-NEXT: |     |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |   `-NullStmt {{.*}} <line:20:7>
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <line:17:35> 'int' lvalue Var {{.*}} 'ripple.par.block.size' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <line:18:8> 'int' lvalue Var {{.*}} 'ripple.loop.iters' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.loop.iters' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |-BinaryOperator {{.*}} <col:8, col:17> 'int' '='
//CHECK-NEXT: |     | |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     | `-BinaryOperator {{.*}} <col:8, col:17> 'int' '+'
//CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   | `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |     |   `-ParenExpr {{.*}} <col:8, col:17> 'int'
//CHECK-NEXT: |     |     `-BinaryOperator {{.*}} <col:8, col:3> 'int' '*'
//CHECK-NEXT: |     |       |-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     |       | `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |     |       `-ImplicitCastExpr {{.*}} <col:3> 'int' <LValueToRValue>
//CHECK-NEXT: |     |         `-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |-ForStmt {{.*}} <col:3, line:20:7>
//CHECK-NEXT: |     | |-<<<NULL>>>
//CHECK-NEXT: |     | |-<<<NULL>>>
//CHECK-NEXT: |     | |-BinaryOperator {{.*}} <line:18:3, col:8> 'int' '<'
//CHECK-NEXT: |     | | |-ImplicitCastExpr {{.*}} <col:3> 'int' <LValueToRValue>
//CHECK-NEXT: |     | | | `-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     | | `-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     | |   `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.loop.iters' 'int'
//CHECK-NEXT: |     | |-CompoundAssignOperator {{.*}} <col:3, col:26> 'int' '+=' ComputeLHSTy='int' ComputeResultTy='int'
//CHECK-NEXT: |     | | |-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     | | `-IntegerLiteral {{.*}} <col:26> 'int' 1
//CHECK-NEXT: |     | `-CompoundStmt {{.*}} <line:19:5, line:20:7>
//CHECK-NEXT: |     |   |-BinaryOperator {{.*}} <line:18:8, col:17> 'int' '='
//CHECK-NEXT: |     |   | |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |   | `-BinaryOperator {{.*}} <col:8, col:17> 'int' '+'
//CHECK-NEXT: |     |   |   |-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |   | `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |     |   |   `-ParenExpr {{.*}} <col:8, col:17> 'int'
//CHECK-NEXT: |     |   |     `-BinaryOperator {{.*}} <col:8, col:3> 'int' '*'
//CHECK-NEXT: |     |   |       |-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |       | `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |     |   |       `-ImplicitCastExpr {{.*}} <col:3> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |         `-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |   `-ForStmt {{.*}} <line:19:5, line:20:7>
//CHECK-NEXT: |     |     |-DeclStmt {{.*}} <line:19:10, col:19>
//CHECK-NEXT: |     |     | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
//CHECK-NEXT: |     |     |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT: |     |     |-<<<NULL>>>
//CHECK-NEXT: |     |     |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
//CHECK-NEXT: |     |     | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
//CHECK-NEXT: |     |     | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |     | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
//CHECK-NEXT: |     |     |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
//CHECK-NEXT: |     |     |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
//CHECK-NEXT: |     |     | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |     `-NullStmt {{.*}} <line:20:7>
//CHECK-NEXT: |     |-BinaryOperator {{.*}} <line:18:3, col:8> 'int' '!='
//CHECK-NEXT: |     | |-BinaryOperator {{.*}} <col:3, line:17:35> 'int' '*'
//CHECK-NEXT: |     | | |-ImplicitCastExpr {{.*}} <line:18:3> 'int' <LValueToRValue>
//CHECK-NEXT: |     | | | `-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     | | `-ImplicitCastExpr {{.*}} <line:17:35> 'int' <LValueToRValue>
//CHECK-NEXT: |     | |   `-DeclRefExpr {{.*}} <col:35> 'int' lvalue Var {{.*}} 'ripple.par.block.size' 'int'
//CHECK-NEXT: |     | `-ImplicitCastExpr {{.*}} <line:18:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.loop.iters' 'int'
//CHECK-NEXT: |     |-BinaryOperator {{.*}} <col:8, col:26> 'int' '='
//CHECK-NEXT: |     | |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     | `-BinaryOperator {{.*}} <col:16, col:26> 'int' '+'
//CHECK-NEXT: |     |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
//CHECK-NEXT: |     |   `-ParenExpr {{.*}} <col:26> 'int'
//CHECK-NEXT: |     |     `-BinaryOperator {{.*}} <col:3, col:26> 'int' '*'
//CHECK-NEXT: |     |       |-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
//CHECK-NEXT: |     |       | |-ParenExpr {{.*}} <col:3> 'int'
//CHECK-NEXT: |     |       | | `-BinaryOperator {{.*}} <col:23, col:3> 'int' '-'
//CHECK-NEXT: |     |       | |   |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
//CHECK-NEXT: |     |       | |   | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
//CHECK-NEXT: |     |       | |   `-ParenExpr {{.*}} <col:3> 'int'
//CHECK-NEXT: |     |       | |     `-BinaryOperator {{.*}} <col:16, <invalid sloc>> 'int' '+'
//CHECK-NEXT: |     |       | |       |-BinaryOperator {{.*}} <col:16, col:26> 'int' '-'
//CHECK-NEXT: |     |       | |       | |-IntegerLiteral {{.*}} <col:16> 'int' 0
//CHECK-NEXT: |     |       | |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
//CHECK-NEXT: |     |       | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
//CHECK-NEXT: |     |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
//CHECK-NEXT: |     |       `-IntegerLiteral {{.*}} <col:26> 'int' 1
//CHECK-NEXT: |     `-ForStmt {{.*}} <line:19:5, line:20:7>
//CHECK-NEXT: |       |-DeclStmt {{.*}} <line:19:10, col:19>
//CHECK-NEXT: |       | `-VarDecl {{.*}} <col:10, col:18> col:14 implicit used i 'int' cinit
//CHECK-NEXT: |       |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT: |       |-<<<NULL>>>
//CHECK-NEXT: |       |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
//CHECK-NEXT: |       | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
//CHECK-NEXT: |       | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |       | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
//CHECK-NEXT: |       |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
//CHECK-NEXT: |       |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
//CHECK-NEXT: |       | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |       `-NullStmt {{.*}} <line:20:7>
//CHECK-NEXT: |-FunctionDecl {{.*}} <line:23:1, line:30:1> line:23:6 test_three 'void (int, int)'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <col:17, col:21> col:21 used x 'int'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <col:24, col:28> col:28 used y 'int'
//CHECK-NEXT: | `-CompoundStmt {{.*}} <col:31, line:30:1>
//CHECK-NEXT: |   |-DeclStmt {{.*}} <line:24:3, col:51>
//CHECK-NEXT: |   | `-VarDecl {{.*}} <col:3, line:4:88> line:24:18 used BS 'ripple_block_t':'struct ripple_block_shape *' cinit
//CHECK-NEXT: |   |   `-CStyleCastExpr {{.*}} <line:4:3, col:88> 'ripple_block_t':'struct ripple_block_shape *' <BitCast>
//CHECK-NEXT: |   |     `-CallExpr {{.*}} <col:20, col:88> 'void *'
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:20> 'void *(*)(__size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t)' <BuiltinFnToFnPtr>
//CHECK-NEXT: |   |       | `-DeclRefExpr {{.*}} <col:20> '<builtin fn type>' Function {{.*}} '__builtin_ripple_set_shape' 'void *(__size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t)'
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:47, col:52> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-ParenExpr {{.*}} <col:47, col:52> 'int'
//CHECK-NEXT: |   |       |   `-IntegerLiteral {{.*}} <line:24:46> 'int' 0
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <line:4:55, col:60> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-ParenExpr {{.*}} <col:55, col:60> 'int'
//CHECK-NEXT: |   |       |   `-IntegerLiteral {{.*}} <line:24:49> 'int' 4
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <line:4:63> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:63> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:66> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:66> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:69> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:69> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:72> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:72> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:75> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:75> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:78> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:78> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:81> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:81> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:84> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:84> 'int' 1
//CHECK-NEXT: |   |       `-ImplicitCastExpr {{.*}} <col:87> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |         `-IntegerLiteral {{.*}} <col:87> 'int' 1
//CHECK-NEXT: |   `-RippleComputeConstruct {{.*}} <line:25:1, col:42>
//CHECK-NEXT: |     |-ForStmt {{.*}} <line:26:3, line:27:42>
//CHECK-NEXT: |     | |-DeclStmt {{.*}} <line:26:8, col:17>
//CHECK-NEXT: |     | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
//CHECK-NEXT: |     | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
//CHECK-NEXT: |     | |-<<<NULL>>>
//CHECK-NEXT: |     | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
//CHECK-NEXT: |     | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
//CHECK-NEXT: |     | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
//CHECK-NEXT: |     | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
//CHECK-NEXT: |     | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
//CHECK-NEXT: |     | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     | `-RippleComputeConstruct {{.*}} <line:27:1, col:42>
//CHECK-NEXT: |     |   |-ForStmt {{.*}} <line:28:5, line:29:7>
//CHECK-NEXT: |     |   | |-DeclStmt {{.*}} <line:28:10, col:19>
//CHECK-NEXT: |     |   | | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
//CHECK-NEXT: |     |   | |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT: |     |   | |-<<<NULL>>>
//CHECK-NEXT: |     |   | |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
//CHECK-NEXT: |     |   | | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   | | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |   | | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   | |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
//CHECK-NEXT: |     |   | |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
//CHECK-NEXT: |     |   | | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |   | `-NullStmt {{.*}} <line:29:7>
//CHECK-NEXT: |     |   |-DeclRefExpr {{.*}} <line:27:35> 'int' lvalue Var {{.*}} 'ripple.par.block.size' 'int'
//CHECK-NEXT: |     |   |-DeclRefExpr {{.*}} <line:28:10> 'int' lvalue Var {{.*}} 'ripple.loop.iters' 'int'
//CHECK-NEXT: |     |   |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.loop.iters' 'int'
//CHECK-NEXT: |     |   |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |     |   |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |     |   |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |   |-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |   |-BinaryOperator {{.*}} <col:10, col:19> 'int' '='
//CHECK-NEXT: |     |   | |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |   | `-BinaryOperator {{.*}} <col:10, col:19> 'int' '+'
//CHECK-NEXT: |     |   |   |-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |   | `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |     |   |   `-ParenExpr {{.*}} <col:10, col:19> 'int'
//CHECK-NEXT: |     |   |     `-BinaryOperator {{.*}} <col:10, col:5> 'int' '*'
//CHECK-NEXT: |     |   |       |-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |       | `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |     |   |       `-ImplicitCastExpr {{.*}} <col:5> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |         `-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |   |-ForStmt {{.*}} <col:5, line:29:7>
//CHECK-NEXT: |     |   | |-<<<NULL>>>
//CHECK-NEXT: |     |   | |-<<<NULL>>>
//CHECK-NEXT: |     |   | |-BinaryOperator {{.*}} <line:28:5, col:10> 'int' '<'
//CHECK-NEXT: |     |   | | |-ImplicitCastExpr {{.*}} <col:5> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   | | | `-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |   | | `-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   | |   `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.loop.iters' 'int'
//CHECK-NEXT: |     |   | |-CompoundAssignOperator {{.*}} <col:5, col:28> 'int' '+=' ComputeLHSTy='int' ComputeResultTy='int'
//CHECK-NEXT: |     |   | | |-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |   | | `-IntegerLiteral {{.*}} <col:28> 'int' 1
//CHECK-NEXT: |     |   | `-CompoundStmt {{.*}} <line:29:7>
//CHECK-NEXT: |     |   |   |-BinaryOperator {{.*}} <line:28:10, col:19> 'int' '='
//CHECK-NEXT: |     |   |   | |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |   |   | `-BinaryOperator {{.*}} <col:10, col:19> 'int' '+'
//CHECK-NEXT: |     |   |   |   |-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |   |   | `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |     |   |   |   `-ParenExpr {{.*}} <col:10, col:19> 'int'
//CHECK-NEXT: |     |   |   |     `-BinaryOperator {{.*}} <col:10, col:5> 'int' '*'
//CHECK-NEXT: |     |   |   |       |-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |   |       | `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |     |   |   |       `-ImplicitCastExpr {{.*}} <col:5> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |   |         `-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |   |   `-NullStmt {{.*}} <line:29:7>
//CHECK-NEXT: |     |   |-BinaryOperator {{.*}} <line:28:5, col:10> 'int' '!='
//CHECK-NEXT: |     |   | |-BinaryOperator {{.*}} <col:5, line:27:35> 'int' '*'
//CHECK-NEXT: |     |   | | |-ImplicitCastExpr {{.*}} <line:28:5> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   | | | `-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |   | | `-ImplicitCastExpr {{.*}} <line:27:35> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   | |   `-DeclRefExpr {{.*}} <col:35> 'int' lvalue Var {{.*}} 'ripple.par.block.size' 'int'
//CHECK-NEXT: |     |   | `-ImplicitCastExpr {{.*}} <line:28:10> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |   `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.loop.iters' 'int'
//CHECK-NEXT: |     |   |-BinaryOperator {{.*}} <col:10, col:28> 'int' '='
//CHECK-NEXT: |     |   | |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |   | `-BinaryOperator {{.*}} <col:18, col:28> 'int' '+'
//CHECK-NEXT: |     |   |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT: |     |   |   `-ParenExpr {{.*}} <col:28> 'int'
//CHECK-NEXT: |     |   |     `-BinaryOperator {{.*}} <col:5, col:28> 'int' '*'
//CHECK-NEXT: |     |   |       |-BinaryOperator {{.*}} <col:5, col:28> 'int' '/'
//CHECK-NEXT: |     |   |       | |-ParenExpr {{.*}} <col:5> 'int'
//CHECK-NEXT: |     |   |       | | `-BinaryOperator {{.*}} <col:25, col:5> 'int' '-'
//CHECK-NEXT: |     |   |       | |   |-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |       | |   | `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
//CHECK-NEXT: |     |   |       | |   `-ParenExpr {{.*}} <col:5> 'int'
//CHECK-NEXT: |     |   |       | |     `-BinaryOperator {{.*}} <col:18, <invalid sloc>> 'int' '+'
//CHECK-NEXT: |     |   |       | |       |-BinaryOperator {{.*}} <col:18, col:28> 'int' '-'
//CHECK-NEXT: |     |   |       | |       | |-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT: |     |   |       | |       | `-IntegerLiteral {{.*}} <col:28> 'int' 1
//CHECK-NEXT: |     |   |       | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
//CHECK-NEXT: |     |   |       | `-IntegerLiteral {{.*}} <col:28> 'int' 1
//CHECK-NEXT: |     |   |       `-IntegerLiteral {{.*}} <col:28> 'int' 1
//CHECK-NEXT: |     |   `-NullStmt {{.*}} <line:29:7>
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <line:25:35> 'int' lvalue Var {{.*}} 'ripple.par.block.size' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <line:26:8> 'int' lvalue Var {{.*}} 'ripple.loop.iters' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.loop.iters' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |-BinaryOperator {{.*}} <col:8, col:17> 'int' '='
//CHECK-NEXT: |     | |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     | `-BinaryOperator {{.*}} <col:8, col:17> 'int' '+'
//CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   | `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |     |   `-ParenExpr {{.*}} <col:8, col:17> 'int'
//CHECK-NEXT: |     |     `-BinaryOperator {{.*}} <col:8, col:3> 'int' '*'
//CHECK-NEXT: |     |       |-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     |       | `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |     |       `-ImplicitCastExpr {{.*}} <col:3> 'int' <LValueToRValue>
//CHECK-NEXT: |     |         `-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |-ForStmt {{.*}} <col:3, line:27:42>
//CHECK-NEXT: |     | |-<<<NULL>>>
//CHECK-NEXT: |     | |-<<<NULL>>>
//CHECK-NEXT: |     | |-BinaryOperator {{.*}} <line:26:3, col:8> 'int' '<'
//CHECK-NEXT: |     | | |-ImplicitCastExpr {{.*}} <col:3> 'int' <LValueToRValue>
//CHECK-NEXT: |     | | | `-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     | | `-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     | |   `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.loop.iters' 'int'
//CHECK-NEXT: |     | |-CompoundAssignOperator {{.*}} <col:3, col:26> 'int' '+=' ComputeLHSTy='int' ComputeResultTy='int'
//CHECK-NEXT: |     | | |-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     | | `-IntegerLiteral {{.*}} <col:26> 'int' 1
//CHECK-NEXT: |     | `-CompoundStmt {{.*}} <line:27:1, col:42>
//CHECK-NEXT: |     |   |-BinaryOperator {{.*}} <line:26:8, col:17> 'int' '='
//CHECK-NEXT: |     |   | |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |   | `-BinaryOperator {{.*}} <col:8, col:17> 'int' '+'
//CHECK-NEXT: |     |   |   |-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |   | `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |     |   |   `-ParenExpr {{.*}} <col:8, col:17> 'int'
//CHECK-NEXT: |     |   |     `-BinaryOperator {{.*}} <col:8, col:3> 'int' '*'
//CHECK-NEXT: |     |   |       |-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |       | `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |     |   |       `-ImplicitCastExpr {{.*}} <col:3> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   |         `-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |   `-RippleComputeConstruct {{.*}} <line:27:1, col:42>
//CHECK-NEXT: |     |     |-ForStmt {{.*}} <line:28:5, line:29:7>
//CHECK-NEXT: |     |     | |-DeclStmt {{.*}} <line:28:10, col:19>
//CHECK-NEXT: |     |     | | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
//CHECK-NEXT: |     |     | |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT: |     |     | |-<<<NULL>>>
//CHECK-NEXT: |     |     | |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
//CHECK-NEXT: |     |     | | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
//CHECK-NEXT: |     |     | | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |     | | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
//CHECK-NEXT: |     |     | |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
//CHECK-NEXT: |     |     | |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
//CHECK-NEXT: |     |     | | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |     | `-NullStmt {{.*}} <line:29:7>
//CHECK-NEXT: |     |     |-DeclRefExpr {{.*}} <line:27:35> 'int' lvalue Var {{.*}} 'ripple.par.block.size' 'int'
//CHECK-NEXT: |     |     |-DeclRefExpr {{.*}} <line:28:10> 'int' lvalue Var {{.*}} 'ripple.loop.iters' 'int'
//CHECK-NEXT: |     |     |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.loop.iters' 'int'
//CHECK-NEXT: |     |     |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |     |     |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |     |     |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |     |-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |     |-BinaryOperator {{.*}} <col:10, col:19> 'int' '='
//CHECK-NEXT: |     |     | |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |     | `-BinaryOperator {{.*}} <col:10, col:19> 'int' '+'
//CHECK-NEXT: |     |     |   |-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |     |     |   | `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |     |     |   `-ParenExpr {{.*}} <col:10, col:19> 'int'
//CHECK-NEXT: |     |     |     `-BinaryOperator {{.*}} <col:10, col:5> 'int' '*'
//CHECK-NEXT: |     |     |       |-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |     |     |       | `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |     |     |       `-ImplicitCastExpr {{.*}} <col:5> 'int' <LValueToRValue>
//CHECK-NEXT: |     |     |         `-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |     |-ForStmt {{.*}} <col:5, line:29:7>
//CHECK-NEXT: |     |     | |-<<<NULL>>>
//CHECK-NEXT: |     |     | |-<<<NULL>>>
//CHECK-NEXT: |     |     | |-BinaryOperator {{.*}} <line:28:5, col:10> 'int' '<'
//CHECK-NEXT: |     |     | | |-ImplicitCastExpr {{.*}} <col:5> 'int' <LValueToRValue>
//CHECK-NEXT: |     |     | | | `-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |     | | `-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |     |     | |   `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.loop.iters' 'int'
//CHECK-NEXT: |     |     | |-CompoundAssignOperator {{.*}} <col:5, col:28> 'int' '+=' ComputeLHSTy='int' ComputeResultTy='int'
//CHECK-NEXT: |     |     | | |-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |     | | `-IntegerLiteral {{.*}} <col:28> 'int' 1
//CHECK-NEXT: |     |     | `-CompoundStmt {{.*}} <line:29:7>
//CHECK-NEXT: |     |     |   |-BinaryOperator {{.*}} <line:28:10, col:19> 'int' '='
//CHECK-NEXT: |     |     |   | |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |     |   | `-BinaryOperator {{.*}} <col:10, col:19> 'int' '+'
//CHECK-NEXT: |     |     |   |   |-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |     |     |   |   | `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |     |     |   |   `-ParenExpr {{.*}} <col:10, col:19> 'int'
//CHECK-NEXT: |     |     |   |     `-BinaryOperator {{.*}} <col:10, col:5> 'int' '*'
//CHECK-NEXT: |     |     |   |       |-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |     |     |   |       | `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |     |     |   |       `-ImplicitCastExpr {{.*}} <col:5> 'int' <LValueToRValue>
//CHECK-NEXT: |     |     |   |         `-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |     |   `-NullStmt {{.*}} <line:29:7>
//CHECK-NEXT: |     |     |-BinaryOperator {{.*}} <line:28:5, col:10> 'int' '!='
//CHECK-NEXT: |     |     | |-BinaryOperator {{.*}} <col:5, line:27:35> 'int' '*'
//CHECK-NEXT: |     |     | | |-ImplicitCastExpr {{.*}} <line:28:5> 'int' <LValueToRValue>
//CHECK-NEXT: |     |     | | | `-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     |     | | `-ImplicitCastExpr {{.*}} <line:27:35> 'int' <LValueToRValue>
//CHECK-NEXT: |     |     | |   `-DeclRefExpr {{.*}} <col:35> 'int' lvalue Var {{.*}} 'ripple.par.block.size' 'int'
//CHECK-NEXT: |     |     | `-ImplicitCastExpr {{.*}} <line:28:10> 'int' <LValueToRValue>
//CHECK-NEXT: |     |     |   `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.loop.iters' 'int'
//CHECK-NEXT: |     |     |-BinaryOperator {{.*}} <col:10, col:28> 'int' '='
//CHECK-NEXT: |     |     | |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     |     | `-BinaryOperator {{.*}} <col:18, col:28> 'int' '+'
//CHECK-NEXT: |     |     |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT: |     |     |   `-ParenExpr {{.*}} <col:28> 'int'
//CHECK-NEXT: |     |     |     `-BinaryOperator {{.*}} <col:5, col:28> 'int' '*'
//CHECK-NEXT: |     |     |       |-BinaryOperator {{.*}} <col:5, col:28> 'int' '/'
//CHECK-NEXT: |     |     |       | |-ParenExpr {{.*}} <col:5> 'int'
//CHECK-NEXT: |     |     |       | | `-BinaryOperator {{.*}} <col:25, col:5> 'int' '-'
//CHECK-NEXT: |     |     |       | |   |-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
//CHECK-NEXT: |     |     |       | |   | `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
//CHECK-NEXT: |     |     |       | |   `-ParenExpr {{.*}} <col:5> 'int'
//CHECK-NEXT: |     |     |       | |     `-BinaryOperator {{.*}} <col:18, <invalid sloc>> 'int' '+'
//CHECK-NEXT: |     |     |       | |       |-BinaryOperator {{.*}} <col:18, col:28> 'int' '-'
//CHECK-NEXT: |     |     |       | |       | |-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT: |     |     |       | |       | `-IntegerLiteral {{.*}} <col:28> 'int' 1
//CHECK-NEXT: |     |     |       | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
//CHECK-NEXT: |     |     |       | `-IntegerLiteral {{.*}} <col:28> 'int' 1
//CHECK-NEXT: |     |     |       `-IntegerLiteral {{.*}} <col:28> 'int' 1
//CHECK-NEXT: |     |     `-NullStmt {{.*}} <line:29:7>
//CHECK-NEXT: |     |-BinaryOperator {{.*}} <line:26:3, col:8> 'int' '!='
//CHECK-NEXT: |     | |-BinaryOperator {{.*}} <col:3, line:25:35> 'int' '*'
//CHECK-NEXT: |     | | |-ImplicitCastExpr {{.*}} <line:26:3> 'int' <LValueToRValue>
//CHECK-NEXT: |     | | | `-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |     | | `-ImplicitCastExpr {{.*}} <line:25:35> 'int' <LValueToRValue>
//CHECK-NEXT: |     | |   `-DeclRefExpr {{.*}} <col:35> 'int' lvalue Var {{.*}} 'ripple.par.block.size' 'int'
//CHECK-NEXT: |     | `-ImplicitCastExpr {{.*}} <line:26:8> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.loop.iters' 'int'
//CHECK-NEXT: |     |-BinaryOperator {{.*}} <col:8, col:26> 'int' '='
//CHECK-NEXT: |     | |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     | `-BinaryOperator {{.*}} <col:16, col:26> 'int' '+'
//CHECK-NEXT: |     |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
//CHECK-NEXT: |     |   `-ParenExpr {{.*}} <col:26> 'int'
//CHECK-NEXT: |     |     `-BinaryOperator {{.*}} <col:3, col:26> 'int' '*'
//CHECK-NEXT: |     |       |-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
//CHECK-NEXT: |     |       | |-ParenExpr {{.*}} <col:3> 'int'
//CHECK-NEXT: |     |       | | `-BinaryOperator {{.*}} <col:23, col:3> 'int' '-'
//CHECK-NEXT: |     |       | |   |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
//CHECK-NEXT: |     |       | |   | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
//CHECK-NEXT: |     |       | |   `-ParenExpr {{.*}} <col:3> 'int'
//CHECK-NEXT: |     |       | |     `-BinaryOperator {{.*}} <col:16, <invalid sloc>> 'int' '+'
//CHECK-NEXT: |     |       | |       |-BinaryOperator {{.*}} <col:16, col:26> 'int' '-'
//CHECK-NEXT: |     |       | |       | |-IntegerLiteral {{.*}} <col:16> 'int' 0
//CHECK-NEXT: |     |       | |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
//CHECK-NEXT: |     |       | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
//CHECK-NEXT: |     |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
//CHECK-NEXT: |     |       `-IntegerLiteral {{.*}} <col:26> 'int' 1
//CHECK-NEXT: |     `-RippleComputeConstruct {{.*}} <line:27:1, col:42>
//CHECK-NEXT: |       |-ForStmt {{.*}} <line:28:5, line:29:7>
//CHECK-NEXT: |       | |-DeclStmt {{.*}} <line:28:10, col:19>
//CHECK-NEXT: |       | | `-VarDecl {{.*}} <col:10, col:18> col:14 implicit used i 'int' cinit
//CHECK-NEXT: |       | |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT: |       | |-<<<NULL>>>
//CHECK-NEXT: |       | |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
//CHECK-NEXT: |       | | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
//CHECK-NEXT: |       | | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |       | | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
//CHECK-NEXT: |       | |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
//CHECK-NEXT: |       | |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
//CHECK-NEXT: |       | | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |       | `-NullStmt {{.*}} <line:29:7>
//CHECK-NEXT: |       |-DeclRefExpr {{.*}} <line:27:35> 'int' lvalue Var {{.*}} 'ripple.par.block.size' 'int'
//CHECK-NEXT: |       |-DeclRefExpr {{.*}} <line:28:10> 'int' lvalue Var {{.*}} 'ripple.loop.iters' 'int'
//CHECK-NEXT: |       |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.loop.iters' 'int'
//CHECK-NEXT: |       |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |       |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |       |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |       |-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |       |-BinaryOperator {{.*}} <col:10, col:19> 'int' '='
//CHECK-NEXT: |       | |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |       | `-BinaryOperator {{.*}} <col:10, col:19> 'int' '+'
//CHECK-NEXT: |       |   |-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |       |   | `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |       |   `-ParenExpr {{.*}} <col:10, col:19> 'int'
//CHECK-NEXT: |       |     `-BinaryOperator {{.*}} <col:10, col:5> 'int' '*'
//CHECK-NEXT: |       |       |-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |       |       | `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |       |       `-ImplicitCastExpr {{.*}} <col:5> 'int' <LValueToRValue>
//CHECK-NEXT: |       |         `-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |       |-ForStmt {{.*}} <col:5, line:29:7>
//CHECK-NEXT: |       | |-<<<NULL>>>
//CHECK-NEXT: |       | |-<<<NULL>>>
//CHECK-NEXT: |       | |-BinaryOperator {{.*}} <line:28:5, col:10> 'int' '<'
//CHECK-NEXT: |       | | |-ImplicitCastExpr {{.*}} <col:5> 'int' <LValueToRValue>
//CHECK-NEXT: |       | | | `-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |       | | `-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |       | |   `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.loop.iters' 'int'
//CHECK-NEXT: |       | |-CompoundAssignOperator {{.*}} <col:5, col:28> 'int' '+=' ComputeLHSTy='int' ComputeResultTy='int'
//CHECK-NEXT: |       | | |-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |       | | `-IntegerLiteral {{.*}} <col:28> 'int' 1
//CHECK-NEXT: |       | `-CompoundStmt {{.*}} <line:29:7>
//CHECK-NEXT: |       |   |-BinaryOperator {{.*}} <line:28:10, col:19> 'int' '='
//CHECK-NEXT: |       |   | |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |       |   | `-BinaryOperator {{.*}} <col:10, col:19> 'int' '+'
//CHECK-NEXT: |       |   |   |-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |       |   |   | `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |       |   |   `-ParenExpr {{.*}} <col:10, col:19> 'int'
//CHECK-NEXT: |       |   |     `-BinaryOperator {{.*}} <col:10, col:5> 'int' '*'
//CHECK-NEXT: |       |   |       |-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |       |   |       | `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |       |   |       `-ImplicitCastExpr {{.*}} <col:5> 'int' <LValueToRValue>
//CHECK-NEXT: |       |   |         `-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |       |   `-NullStmt {{.*}} <line:29:7>
//CHECK-NEXT: |       |-BinaryOperator {{.*}} <line:28:5, col:10> 'int' '!='
//CHECK-NEXT: |       | |-BinaryOperator {{.*}} <col:5, line:27:35> 'int' '*'
//CHECK-NEXT: |       | | |-ImplicitCastExpr {{.*}} <line:28:5> 'int' <LValueToRValue>
//CHECK-NEXT: |       | | | `-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |       | | `-ImplicitCastExpr {{.*}} <line:27:35> 'int' <LValueToRValue>
//CHECK-NEXT: |       | |   `-DeclRefExpr {{.*}} <col:35> 'int' lvalue Var {{.*}} 'ripple.par.block.size' 'int'
//CHECK-NEXT: |       | `-ImplicitCastExpr {{.*}} <line:28:10> 'int' <LValueToRValue>
//CHECK-NEXT: |       |   `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.loop.iters' 'int'
//CHECK-NEXT: |       |-BinaryOperator {{.*}} <col:10, col:28> 'int' '='
//CHECK-NEXT: |       | |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |       | `-BinaryOperator {{.*}} <col:18, col:28> 'int' '+'
//CHECK-NEXT: |       |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT: |       |   `-ParenExpr {{.*}} <col:28> 'int'
//CHECK-NEXT: |       |     `-BinaryOperator {{.*}} <col:5, col:28> 'int' '*'
//CHECK-NEXT: |       |       |-BinaryOperator {{.*}} <col:5, col:28> 'int' '/'
//CHECK-NEXT: |       |       | |-ParenExpr {{.*}} <col:5> 'int'
//CHECK-NEXT: |       |       | | `-BinaryOperator {{.*}} <col:25, col:5> 'int' '-'
//CHECK-NEXT: |       |       | |   |-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
//CHECK-NEXT: |       |       | |   | `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
//CHECK-NEXT: |       |       | |   `-ParenExpr {{.*}} <col:5> 'int'
//CHECK-NEXT: |       |       | |     `-BinaryOperator {{.*}} <col:18, <invalid sloc>> 'int' '+'
//CHECK-NEXT: |       |       | |       |-BinaryOperator {{.*}} <col:18, col:28> 'int' '-'
//CHECK-NEXT: |       |       | |       | |-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT: |       |       | |       | `-IntegerLiteral {{.*}} <col:28> 'int' 1
//CHECK-NEXT: |       |       | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
//CHECK-NEXT: |       |       | `-IntegerLiteral {{.*}} <col:28> 'int' 1
//CHECK-NEXT: |       |       `-IntegerLiteral {{.*}} <col:28> 'int' 1
//CHECK-NEXT: |       `-NullStmt {{.*}} <line:29:7>
//CHECK-NEXT: |-FunctionDecl {{.*}} <line:32:1, line:38:1> line:32:6 test_four 'void (int, int)'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <col:16, col:20> col:20 used x 'int'
//CHECK-NEXT: | |-ParmVarDecl {{.*}} <col:23, col:27> col:27 used y 'int'
//CHECK-NEXT: | `-CompoundStmt {{.*}} <col:30, line:38:1>
//CHECK-NEXT: |   |-DeclStmt {{.*}} <line:33:3, col:51>
//CHECK-NEXT: |   | `-VarDecl {{.*}} <col:3, line:4:88> line:33:18 used BS 'ripple_block_t':'struct ripple_block_shape *' cinit
//CHECK-NEXT: |   |   `-CStyleCastExpr {{.*}} <line:4:3, col:88> 'ripple_block_t':'struct ripple_block_shape *' <BitCast>
//CHECK-NEXT: |   |     `-CallExpr {{.*}} <col:20, col:88> 'void *'
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:20> 'void *(*)(__size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t)' <BuiltinFnToFnPtr>
//CHECK-NEXT: |   |       | `-DeclRefExpr {{.*}} <col:20> '<builtin fn type>' Function {{.*}} '__builtin_ripple_set_shape' 'void *(__size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t)'
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:47, col:52> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-ParenExpr {{.*}} <col:47, col:52> 'int'
//CHECK-NEXT: |   |       |   `-IntegerLiteral {{.*}} <line:33:46> 'int' 0
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <line:4:55, col:60> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-ParenExpr {{.*}} <col:55, col:60> 'int'
//CHECK-NEXT: |   |       |   `-IntegerLiteral {{.*}} <line:33:49> 'int' 4
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <line:4:63> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:63> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:66> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:66> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:69> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:69> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:72> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:72> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:75> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:75> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:78> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:78> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:81> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:81> 'int' 1
//CHECK-NEXT: |   |       |-ImplicitCastExpr {{.*}} <col:84> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |       | `-IntegerLiteral {{.*}} <col:84> 'int' 1
//CHECK-NEXT: |   |       `-ImplicitCastExpr {{.*}} <col:87> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT: |   |         `-IntegerLiteral {{.*}} <col:87> 'int' 1
//CHECK-NEXT: |   `-ForStmt {{.*}} <line:34:3, line:35:42>
//CHECK-NEXT: |     |-DeclStmt {{.*}} <line:34:8, col:17>
//CHECK-NEXT: |     | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
//CHECK-NEXT: |     |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
//CHECK-NEXT: |     |-<<<NULL>>>
//CHECK-NEXT: |     |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
//CHECK-NEXT: |     | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
//CHECK-NEXT: |     | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
//CHECK-NEXT: |     |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
//CHECK-NEXT: |     |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
//CHECK-NEXT: |     | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |     `-RippleComputeConstruct {{.*}} <line:35:1, col:42>
//CHECK-NEXT: |       |-ForStmt {{.*}} <line:36:5, line:37:7>
//CHECK-NEXT: |       | |-DeclStmt {{.*}} <line:36:10, col:19>
//CHECK-NEXT: |       | | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
//CHECK-NEXT: |       | |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT: |       | |-<<<NULL>>>
//CHECK-NEXT: |       | |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
//CHECK-NEXT: |       | | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
//CHECK-NEXT: |       | | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |       | | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
//CHECK-NEXT: |       | |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
//CHECK-NEXT: |       | |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
//CHECK-NEXT: |       | | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |       | `-NullStmt {{.*}} <line:37:7>
//CHECK-NEXT: |       |-DeclRefExpr {{.*}} <line:35:35> 'int' lvalue Var {{.*}} 'ripple.par.block.size' 'int'
//CHECK-NEXT: |       |-DeclRefExpr {{.*}} <line:36:10> 'int' lvalue Var {{.*}} 'ripple.loop.iters' 'int'
//CHECK-NEXT: |       |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.loop.iters' 'int'
//CHECK-NEXT: |       |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |       |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |       |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |       |-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |       |-BinaryOperator {{.*}} <col:10, col:19> 'int' '='
//CHECK-NEXT: |       | |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |       | `-BinaryOperator {{.*}} <col:10, col:19> 'int' '+'
//CHECK-NEXT: |       |   |-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |       |   | `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |       |   `-ParenExpr {{.*}} <col:10, col:19> 'int'
//CHECK-NEXT: |       |     `-BinaryOperator {{.*}} <col:10, col:5> 'int' '*'
//CHECK-NEXT: |       |       |-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |       |       | `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |       |       `-ImplicitCastExpr {{.*}} <col:5> 'int' <LValueToRValue>
//CHECK-NEXT: |       |         `-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |       |-ForStmt {{.*}} <col:5, line:37:7>
//CHECK-NEXT: |       | |-<<<NULL>>>
//CHECK-NEXT: |       | |-<<<NULL>>>
//CHECK-NEXT: |       | |-BinaryOperator {{.*}} <line:36:5, col:10> 'int' '<'
//CHECK-NEXT: |       | | |-ImplicitCastExpr {{.*}} <col:5> 'int' <LValueToRValue>
//CHECK-NEXT: |       | | | `-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |       | | `-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |       | |   `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.loop.iters' 'int'
//CHECK-NEXT: |       | |-CompoundAssignOperator {{.*}} <col:5, col:28> 'int' '+=' ComputeLHSTy='int' ComputeResultTy='int'
//CHECK-NEXT: |       | | |-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |       | | `-IntegerLiteral {{.*}} <col:28> 'int' 1
//CHECK-NEXT: |       | `-CompoundStmt {{.*}} <line:37:7>
//CHECK-NEXT: |       |   |-BinaryOperator {{.*}} <line:36:10, col:19> 'int' '='
//CHECK-NEXT: |       |   | |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |       |   | `-BinaryOperator {{.*}} <col:10, col:19> 'int' '+'
//CHECK-NEXT: |       |   |   |-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |       |   |   | `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT: |       |   |   `-ParenExpr {{.*}} <col:10, col:19> 'int'
//CHECK-NEXT: |       |   |     `-BinaryOperator {{.*}} <col:10, col:5> 'int' '*'
//CHECK-NEXT: |       |   |       |-ImplicitCastExpr {{.*}} <col:10> 'int' <LValueToRValue>
//CHECK-NEXT: |       |   |       | `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT: |       |   |       `-ImplicitCastExpr {{.*}} <col:5> 'int' <LValueToRValue>
//CHECK-NEXT: |       |   |         `-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |       |   `-NullStmt {{.*}} <line:37:7>
//CHECK-NEXT: |       |-BinaryOperator {{.*}} <line:36:5, col:10> 'int' '!='
//CHECK-NEXT: |       | |-BinaryOperator {{.*}} <col:5, line:35:35> 'int' '*'
//CHECK-NEXT: |       | | |-ImplicitCastExpr {{.*}} <line:36:5> 'int' <LValueToRValue>
//CHECK-NEXT: |       | | | `-DeclRefExpr {{.*}} <col:5> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT: |       | | `-ImplicitCastExpr {{.*}} <line:35:35> 'int' <LValueToRValue>
//CHECK-NEXT: |       | |   `-DeclRefExpr {{.*}} <col:35> 'int' lvalue Var {{.*}} 'ripple.par.block.size' 'int'
//CHECK-NEXT: |       | `-ImplicitCastExpr {{.*}} <line:36:10> 'int' <LValueToRValue>
//CHECK-NEXT: |       |   `-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'ripple.loop.iters' 'int'
//CHECK-NEXT: |       |-BinaryOperator {{.*}} <col:10, col:28> 'int' '='
//CHECK-NEXT: |       | |-DeclRefExpr {{.*}} <col:10> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT: |       | `-BinaryOperator {{.*}} <col:18, col:28> 'int' '+'
//CHECK-NEXT: |       |   |-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT: |       |   `-ParenExpr {{.*}} <col:28> 'int'
//CHECK-NEXT: |       |     `-BinaryOperator {{.*}} <col:5, col:28> 'int' '*'
//CHECK-NEXT: |       |       |-BinaryOperator {{.*}} <col:5, col:28> 'int' '/'
//CHECK-NEXT: |       |       | |-ParenExpr {{.*}} <col:5> 'int'
//CHECK-NEXT: |       |       | | `-BinaryOperator {{.*}} <col:25, col:5> 'int' '-'
//CHECK-NEXT: |       |       | |   |-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
//CHECK-NEXT: |       |       | |   | `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
//CHECK-NEXT: |       |       | |   `-ParenExpr {{.*}} <col:5> 'int'
//CHECK-NEXT: |       |       | |     `-BinaryOperator {{.*}} <col:18, <invalid sloc>> 'int' '+'
//CHECK-NEXT: |       |       | |       |-BinaryOperator {{.*}} <col:18, col:28> 'int' '-'
//CHECK-NEXT: |       |       | |       | |-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT: |       |       | |       | `-IntegerLiteral {{.*}} <col:28> 'int' 1
//CHECK-NEXT: |       |       | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
//CHECK-NEXT: |       |       | `-IntegerLiteral {{.*}} <col:28> 'int' 1
//CHECK-NEXT: |       |       `-IntegerLiteral {{.*}} <col:28> 'int' 1
//CHECK-NEXT: |       `-NullStmt {{.*}} <line:37:7>
//CHECK-NEXT: `-FunctionDecl {{.*}} <line:40:1, line:47:1> line:40:6 test_five 'void (int, int, int)'
//CHECK-NEXT:   |-ParmVarDecl {{.*}} <col:16, col:20> col:20 used x 'int'
//CHECK-NEXT:   |-ParmVarDecl {{.*}} <col:23, col:27> col:27 used y 'int'
//CHECK-NEXT:   |-ParmVarDecl {{.*}} <col:30, col:34> col:34 used z 'int'
//CHECK-NEXT:   `-CompoundStmt {{.*}} <col:37, line:47:1>
//CHECK-NEXT:     |-DeclStmt {{.*}} <line:41:3, col:51>
//CHECK-NEXT:     | `-VarDecl {{.*}} <col:3, line:4:88> line:41:18 used BS 'ripple_block_t':'struct ripple_block_shape *' cinit
//CHECK-NEXT:     |   `-CStyleCastExpr {{.*}} <line:4:3, col:88> 'ripple_block_t':'struct ripple_block_shape *' <BitCast>
//CHECK-NEXT:     |     `-CallExpr {{.*}} <col:20, col:88> 'void *'
//CHECK-NEXT:     |       |-ImplicitCastExpr {{.*}} <col:20> 'void *(*)(__size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t)' <BuiltinFnToFnPtr>
//CHECK-NEXT:     |       | `-DeclRefExpr {{.*}} <col:20> '<builtin fn type>' Function {{.*}} '__builtin_ripple_set_shape' 'void *(__size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t, __size_t)'
//CHECK-NEXT:     |       |-ImplicitCastExpr {{.*}} <col:47, col:52> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT:     |       | `-ParenExpr {{.*}} <col:47, col:52> 'int'
//CHECK-NEXT:     |       |   `-IntegerLiteral {{.*}} <line:41:46> 'int' 0
//CHECK-NEXT:     |       |-ImplicitCastExpr {{.*}} <line:4:55, col:60> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT:     |       | `-ParenExpr {{.*}} <col:55, col:60> 'int'
//CHECK-NEXT:     |       |   `-IntegerLiteral {{.*}} <line:41:49> 'int' 4
//CHECK-NEXT:     |       |-ImplicitCastExpr {{.*}} <line:4:63> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT:     |       | `-IntegerLiteral {{.*}} <col:63> 'int' 1
//CHECK-NEXT:     |       |-ImplicitCastExpr {{.*}} <col:66> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT:     |       | `-IntegerLiteral {{.*}} <col:66> 'int' 1
//CHECK-NEXT:     |       |-ImplicitCastExpr {{.*}} <col:69> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT:     |       | `-IntegerLiteral {{.*}} <col:69> 'int' 1
//CHECK-NEXT:     |       |-ImplicitCastExpr {{.*}} <col:72> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT:     |       | `-IntegerLiteral {{.*}} <col:72> 'int' 1
//CHECK-NEXT:     |       |-ImplicitCastExpr {{.*}} <col:75> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT:     |       | `-IntegerLiteral {{.*}} <col:75> 'int' 1
//CHECK-NEXT:     |       |-ImplicitCastExpr {{.*}} <col:78> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT:     |       | `-IntegerLiteral {{.*}} <col:78> 'int' 1
//CHECK-NEXT:     |       |-ImplicitCastExpr {{.*}} <col:81> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT:     |       | `-IntegerLiteral {{.*}} <col:81> 'int' 1
//CHECK-NEXT:     |       |-ImplicitCastExpr {{.*}} <col:84> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT:     |       | `-IntegerLiteral {{.*}} <col:84> 'int' 1
//CHECK-NEXT:     |       `-ImplicitCastExpr {{.*}} <col:87> '__size_t':'unsigned long' <IntegralCast>
//CHECK-NEXT:     |         `-IntegerLiteral {{.*}} <col:87> 'int' 1
//CHECK-NEXT:     `-RippleComputeConstruct {{.*}} <line:42:1, col:45>
//CHECK-NEXT:       |-ForStmt {{.*}} <line:43:3, line:46:9>
//CHECK-NEXT:       | |-DeclStmt {{.*}} <line:43:8, col:17>
//CHECK-NEXT:       | | `-VarDecl {{.*}} <col:8, col:16> col:12 used i 'int' cinit
//CHECK-NEXT:       | |   `-IntegerLiteral {{.*}} <col:16> 'int' 0
//CHECK-NEXT:       | |-<<<NULL>>>
//CHECK-NEXT:       | |-BinaryOperator {{.*}} <col:19, col:23> 'int' '<'
//CHECK-NEXT:       | | |-ImplicitCastExpr {{.*}} <col:19> 'int' <LValueToRValue>
//CHECK-NEXT:       | | | `-DeclRefExpr {{.*}} <col:19> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:       | | `-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
//CHECK-NEXT:       | |   `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
//CHECK-NEXT:       | |-UnaryOperator {{.*}} <col:26, col:27> 'int' postfix '++'
//CHECK-NEXT:       | | `-DeclRefExpr {{.*}} <col:26> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:       | `-ForStmt {{.*}} <line:44:5, line:46:9>
//CHECK-NEXT:       |   |-DeclStmt {{.*}} <line:44:10, col:19>
//CHECK-NEXT:       |   | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
//CHECK-NEXT:       |   |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT:       |   |-<<<NULL>>>
//CHECK-NEXT:       |   |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
//CHECK-NEXT:       |   | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
//CHECK-NEXT:       |   | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:       |   | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
//CHECK-NEXT:       |   |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
//CHECK-NEXT:       |   |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
//CHECK-NEXT:       |   | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:       |   `-ForStmt {{.*}} <line:45:7, line:46:9>
//CHECK-NEXT:       |     |-DeclStmt {{.*}} <line:45:12, col:21>
//CHECK-NEXT:       |     | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
//CHECK-NEXT:       |     |   `-IntegerLiteral {{.*}} <col:20> 'int' 0
//CHECK-NEXT:       |     |-<<<NULL>>>
//CHECK-NEXT:       |     |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
//CHECK-NEXT:       |     | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
//CHECK-NEXT:       |     | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:       |     | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
//CHECK-NEXT:       |     |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
//CHECK-NEXT:       |     |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
//CHECK-NEXT:       |     | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:       |     `-NullStmt {{.*}} <line:46:9>
//CHECK-NEXT:       |-DeclRefExpr {{.*}} <line:42:35> 'int' lvalue Var {{.*}} 'ripple.par.block.size' 'int'
//CHECK-NEXT:       |-DeclRefExpr {{.*}} <line:43:8> 'int' lvalue Var {{.*}} 'ripple.loop.iters' 'int'
//CHECK-NEXT:       |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.loop.iters' 'int'
//CHECK-NEXT:       |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT:       |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT:       |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:       |-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT:       |-BinaryOperator {{.*}} <col:8, col:17> 'int' '='
//CHECK-NEXT:       | |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:       | `-BinaryOperator {{.*}} <col:8, col:17> 'int' '+'
//CHECK-NEXT:       |   |-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT:       |   | `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT:       |   `-ParenExpr {{.*}} <col:8, col:17> 'int'
//CHECK-NEXT:       |     `-BinaryOperator {{.*}} <col:8, col:3> 'int' '*'
//CHECK-NEXT:       |       |-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT:       |       | `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT:       |       `-ImplicitCastExpr {{.*}} <col:3> 'int' <LValueToRValue>
//CHECK-NEXT:       |         `-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT:       |-ForStmt {{.*}} <col:3, line:46:9>
//CHECK-NEXT:       | |-<<<NULL>>>
//CHECK-NEXT:       | |-<<<NULL>>>
//CHECK-NEXT:       | |-BinaryOperator {{.*}} <line:43:3, col:8> 'int' '<'
//CHECK-NEXT:       | | |-ImplicitCastExpr {{.*}} <col:3> 'int' <LValueToRValue>
//CHECK-NEXT:       | | | `-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT:       | | `-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT:       | |   `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.loop.iters' 'int'
//CHECK-NEXT:       | |-CompoundAssignOperator {{.*}} <col:3, col:26> 'int' '+=' ComputeLHSTy='int' ComputeResultTy='int'
//CHECK-NEXT:       | | |-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT:       | | `-IntegerLiteral {{.*}} <col:26> 'int' 1
//CHECK-NEXT:       | `-CompoundStmt {{.*}} <line:44:5, line:46:9>
//CHECK-NEXT:       |   |-BinaryOperator {{.*}} <line:43:8, col:17> 'int' '='
//CHECK-NEXT:       |   | |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:       |   | `-BinaryOperator {{.*}} <col:8, col:17> 'int' '+'
//CHECK-NEXT:       |   |   |-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT:       |   |   | `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.init' 'int'
//CHECK-NEXT:       |   |   `-ParenExpr {{.*}} <col:8, col:17> 'int'
//CHECK-NEXT:       |   |     `-BinaryOperator {{.*}} <col:8, col:3> 'int' '*'
//CHECK-NEXT:       |   |       |-ImplicitCastExpr {{.*}} <col:8> 'int' <LValueToRValue>
//CHECK-NEXT:       |   |       | `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.par.step' 'int'
//CHECK-NEXT:       |   |       `-ImplicitCastExpr {{.*}} <col:3> 'int' <LValueToRValue>
//CHECK-NEXT:       |   |         `-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT:       |   `-ForStmt {{.*}} <line:44:5, line:46:9>
//CHECK-NEXT:       |     |-DeclStmt {{.*}} <line:44:10, col:19>
//CHECK-NEXT:       |     | `-VarDecl {{.*}} <col:10, col:18> col:14 used i 'int' cinit
//CHECK-NEXT:       |     |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT:       |     |-<<<NULL>>>
//CHECK-NEXT:       |     |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
//CHECK-NEXT:       |     | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
//CHECK-NEXT:       |     | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:       |     | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
//CHECK-NEXT:       |     |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
//CHECK-NEXT:       |     |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
//CHECK-NEXT:       |     | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:       |     `-ForStmt {{.*}} <line:45:7, line:46:9>
//CHECK-NEXT:       |       |-DeclStmt {{.*}} <line:45:12, col:21>
//CHECK-NEXT:       |       | `-VarDecl {{.*}} <col:12, col:20> col:16 used i 'int' cinit
//CHECK-NEXT:       |       |   `-IntegerLiteral {{.*}} <col:20> 'int' 0
//CHECK-NEXT:       |       |-<<<NULL>>>
//CHECK-NEXT:       |       |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
//CHECK-NEXT:       |       | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
//CHECK-NEXT:       |       | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:       |       | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
//CHECK-NEXT:       |       |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
//CHECK-NEXT:       |       |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
//CHECK-NEXT:       |       | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:       |       `-NullStmt {{.*}} <line:46:9>
//CHECK-NEXT:       |-BinaryOperator {{.*}} <line:43:3, col:8> 'int' '!='
//CHECK-NEXT:       | |-BinaryOperator {{.*}} <col:3, line:42:35> 'int' '*'
//CHECK-NEXT:       | | |-ImplicitCastExpr {{.*}} <line:43:3> 'int' <LValueToRValue>
//CHECK-NEXT:       | | | `-DeclRefExpr {{.*}} <col:3> 'int' lvalue Var {{.*}} 'ripple.par.iv' 'int'
//CHECK-NEXT:       | | `-ImplicitCastExpr {{.*}} <line:42:35> 'int' <LValueToRValue>
//CHECK-NEXT:       | |   `-DeclRefExpr {{.*}} <col:35> 'int' lvalue Var {{.*}} 'ripple.par.block.size' 'int'
//CHECK-NEXT:       | `-ImplicitCastExpr {{.*}} <line:43:8> 'int' <LValueToRValue>
//CHECK-NEXT:       |   `-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'ripple.loop.iters' 'int'
//CHECK-NEXT:       |-BinaryOperator {{.*}} <col:8, col:26> 'int' '='
//CHECK-NEXT:       | |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:       | `-BinaryOperator {{.*}} <col:16, col:26> 'int' '+'
//CHECK-NEXT:       |   |-IntegerLiteral {{.*}} <col:16> 'int' 0
//CHECK-NEXT:       |   `-ParenExpr {{.*}} <col:26> 'int'
//CHECK-NEXT:       |     `-BinaryOperator {{.*}} <col:3, col:26> 'int' '*'
//CHECK-NEXT:       |       |-BinaryOperator {{.*}} <col:3, col:26> 'int' '/'
//CHECK-NEXT:       |       | |-ParenExpr {{.*}} <col:3> 'int'
//CHECK-NEXT:       |       | | `-BinaryOperator {{.*}} <col:23, col:3> 'int' '-'
//CHECK-NEXT:       |       | |   |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
//CHECK-NEXT:       |       | |   | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue ParmVar {{.*}} 'x' 'int'
//CHECK-NEXT:       |       | |   `-ParenExpr {{.*}} <col:3> 'int'
//CHECK-NEXT:       |       | |     `-BinaryOperator {{.*}} <col:16, <invalid sloc>> 'int' '+'
//CHECK-NEXT:       |       | |       |-BinaryOperator {{.*}} <col:16, col:26> 'int' '-'
//CHECK-NEXT:       |       | |       | |-IntegerLiteral {{.*}} <col:16> 'int' 0
//CHECK-NEXT:       |       | |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
//CHECK-NEXT:       |       | |       `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 1
//CHECK-NEXT:       |       | `-IntegerLiteral {{.*}} <col:26> 'int' 1
//CHECK-NEXT:       |       `-IntegerLiteral {{.*}} <col:26> 'int' 1
//CHECK-NEXT:       `-ForStmt {{.*}} <line:44:5, line:46:9>
//CHECK-NEXT:         |-DeclStmt {{.*}} <line:44:10, col:19>
//CHECK-NEXT:         | `-VarDecl {{.*}} <col:10, col:18> col:14 implicit used i 'int' cinit
//CHECK-NEXT:         |   `-IntegerLiteral {{.*}} <col:18> 'int' 0
//CHECK-NEXT:         |-<<<NULL>>>
//CHECK-NEXT:         |-BinaryOperator {{.*}} <col:21, col:25> 'int' '<'
//CHECK-NEXT:         | |-ImplicitCastExpr {{.*}} <col:21> 'int' <LValueToRValue>
//CHECK-NEXT:         | | `-DeclRefExpr {{.*}} <col:21> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:         | `-ImplicitCastExpr {{.*}} <col:25> 'int' <LValueToRValue>
//CHECK-NEXT:         |   `-DeclRefExpr {{.*}} <col:25> 'int' lvalue ParmVar {{.*}} 'y' 'int'
//CHECK-NEXT:         |-UnaryOperator {{.*}} <col:28, col:29> 'int' postfix '++'
//CHECK-NEXT:         | `-DeclRefExpr {{.*}} <col:28> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:         `-ForStmt {{.*}} <line:45:7, line:46:9>
//CHECK-NEXT:           |-DeclStmt {{.*}} <line:45:12, col:21>
//CHECK-NEXT:           | `-VarDecl {{.*}} <col:12, col:20> col:16 implicit used i 'int' cinit
//CHECK-NEXT:           |   `-IntegerLiteral {{.*}} <col:20> 'int' 0
//CHECK-NEXT:           |-<<<NULL>>>
//CHECK-NEXT:           |-BinaryOperator {{.*}} <col:23, col:27> 'int' '<'
//CHECK-NEXT:           | |-ImplicitCastExpr {{.*}} <col:23> 'int' <LValueToRValue>
//CHECK-NEXT:           | | `-DeclRefExpr {{.*}} <col:23> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:           | `-ImplicitCastExpr {{.*}} <col:27> 'int' <LValueToRValue>
//CHECK-NEXT:           |   `-DeclRefExpr {{.*}} <col:27> 'int' lvalue ParmVar {{.*}} 'z' 'int'
//CHECK-NEXT:           |-UnaryOperator {{.*}} <col:30, col:31> 'int' postfix '++'
//CHECK-NEXT:           | `-DeclRefExpr {{.*}} <col:30> 'int' lvalue Var {{.*}} 'i' 'int'
//CHECK-NEXT:           `-NullStmt {{.*}} <line:46:9>
