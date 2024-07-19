// RUN: %clang_cc1 -std=c++17 -Wno-unused -ast-dump %s -ast-dump-filter Test | FileCheck %s

namespace Test {
  template<typename T, typename U>
  void Unary(T t, T* pt, T U::* mpt, T(&ft)(), T(&at)[4]) {
    // CHECK: UnaryOperator {{.*}} '<dependent type>' lvalue prefix '*' cannot overflow
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T' lvalue ParmVar {{.*}} 't' 'T'
    *t;

    // CHECK: UnaryOperator {{.*}} '<dependent type>' prefix '+' cannot overflow
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T' lvalue ParmVar {{.*}} 't' 'T'
    +t;

    // CHECK: UnaryOperator {{.*}} '<dependent type>' prefix '-' cannot overflow
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T' lvalue ParmVar {{.*}} 't' 'T'
    -t;

    // CHECK: UnaryOperator {{.*}} '<dependent type>' prefix '!' cannot overflow
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T' lvalue ParmVar {{.*}} 't' 'T'
    !t;

    // CHECK: UnaryOperator {{.*}} '<dependent type>' prefix '~' cannot overflow
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T' lvalue ParmVar {{.*}} 't' 'T'
    ~t;

    // CHECK: UnaryOperator {{.*}} '<dependent type>' prefix '&' cannot overflow
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T' lvalue ParmVar {{.*}} 't' 'T'
    &t;

    // CHECK: UnaryOperator {{.*}} '<dependent type>' lvalue prefix '++' cannot overflow
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T' lvalue ParmVar {{.*}} 't' 'T'
    ++t;

    // CHECK: UnaryOperator {{.*}} '<dependent type>' lvalue prefix '--' cannot overflow
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T' lvalue ParmVar {{.*}} 't' 'T'
    --t;

    // CHECK: UnaryOperator {{.*}} 'T' lvalue prefix '*' cannot overflow
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'T *' <LValueToRValue>
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
    *pt;

    // CHECK: UnaryOperator {{.*}} 'T *' prefix '+' cannot overflow
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'T *' <LValueToRValue>
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
    +pt;

    // CHECK: UnaryOperator {{.*}} 'bool' prefix '!' cannot overflow
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'bool' <PointerToBoolean>
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'T *' <LValueToRValue>
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
    !pt;

    // CHECK: UnaryOperator {{.*}} '<dependent type>' prefix '&' cannot overflow
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
    &pt;

    // CHECK: UnaryOperator {{.*}} 'T *' lvalue prefix '++' cannot overflow
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
    ++pt;

    // CHECK: UnaryOperator {{.*}} 'T *' lvalue prefix '--' cannot overflow
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
    --pt;

    // CHECK: UnaryOperator {{.*}} 'bool' prefix '!' cannot overflow
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'bool' <MemberPointerToBoolean>
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'T U::*' <LValueToRValue>
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T U::*' lvalue ParmVar {{.*}} 'mpt' 'T U::*'
    !mpt;

    // CHECK: UnaryOperator {{.*}} '<dependent type>' prefix '&' cannot overflow
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T U::*' lvalue ParmVar {{.*}} 'mpt' 'T U::*'
    &mpt;

    // CHECK: UnaryOperator {{.*}} 'T ()' lvalue prefix '*' cannot overflow
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'T (*)()' <FunctionToPointerDecay>
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T ()' lvalue ParmVar {{.*}} 'ft' 'T (&)()'
    *ft;

    // CHECK: UnaryOperator {{.*}} 'T (*)()' prefix '+' cannot overflow
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'T (*)()' <FunctionToPointerDecay>
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T ()' lvalue ParmVar {{.*}} 'ft' 'T (&)()'
    +ft;

    // CHECK: UnaryOperator {{.*}} 'bool' prefix '!' cannot overflow
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'bool' <PointerToBoolean>
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'T (*)()' <FunctionToPointerDecay>
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T ()' lvalue ParmVar {{.*}} 'ft' 'T (&)()'
    !ft;

    // CHECK: UnaryOperator {{.*}} '<dependent type>' prefix '&' cannot overflow
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T ()' lvalue ParmVar {{.*}} 'ft' 'T (&)()'
    &ft;

    // CHECK: UnaryOperator {{.*}} 'T' lvalue prefix '*' cannot overflow
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'T *' <ArrayToPointerDecay>
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T[4]' lvalue ParmVar {{.*}} 'at' 'T (&)[4]'
    *at;

    // CHECK: UnaryOperator {{.*}} 'T *' prefix '+' cannot overflow
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'T *' <ArrayToPointerDecay>
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T[4]' lvalue ParmVar {{.*}} 'at' 'T (&)[4]'
    +at;

    // CHECK: UnaryOperator {{.*}} 'bool' prefix '!' cannot overflow
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'bool' <PointerToBoolean>
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'T *' <ArrayToPointerDecay>
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T[4]' lvalue ParmVar {{.*}} 'at' 'T (&)[4]'
    !at;

    // CHECK: UnaryOperator {{.*}} '<dependent type>' prefix '&' cannot overflow
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T[4]' lvalue ParmVar {{.*}} 'at' 'T (&)[4]'
    &at;
  }

  template<typename T, typename U>
  void Binary(T* pt, U* pu) {
    // CHECK: BinaryOperator {{.*}} '<dependent type>' '+'
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
    pt + 3;

    // CHECK: BinaryOperator {{.*}} '<dependent type>' '-'
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
    pt - pt;

    // CHECK: BinaryOperator {{.*}} '<dependent type>' '-'
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
    // CHECK-NEXT: DeclRefExpr {{.*}} 'U *' lvalue ParmVar {{.*}} 'pu' 'U *'
    pt - pu;

    // CHECK: BinaryOperator {{.*}} '<dependent type>' '=='
    // CHECK-NEXT: DeclRefExpr {{.*}} 'T *' lvalue ParmVar {{.*}} 'pt' 'T *'
    // CHECK-NEXT: DeclRefExpr {{.*}} 'U *' lvalue ParmVar {{.*}} 'pu' 'U *'
    pt == pu;
  }
} // namespace Test
