// RUN: %clang_cc1 -std=c++23 -triple x86_64-linux-gnu -fcxx-exceptions -ast-dump %s \
// RUN: | FileCheck -strict-whitespace %s

namespace P2718R0 {

// Test basic
struct A {
  int a[3] = {1, 2, 3};
  A() {}
  ~A() {}
  const int *begin() const { return a; }
  const int *end() const { return a + 3; }
  A& r() { return *this; }
  A g() { return A(); }
};

A g() { return A(); }
const A &f1(const A &t) { return t; }

void test1() {
  [[maybe_unused]] int sum = 0;

  for (auto e : f1(g()))
    sum += e;
}

// CHECK: |-FunctionDecl {{.*}} test1 'void ()'
// CHECK-NEXT:   | `-CompoundStmt {{.*}}
// CHECK-NEXT:   |   |-DeclStmt {{.*}}
// CHECK-NEXT:   |   | `-VarDecl {{.*}} used sum 'int' cinit
// CHECK-NEXT:   |   |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT:   |   |   |-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:   |   |   `-attrDetails: UnusedAttr {{.*}} maybe_unused
// CHECK-NEXT:   |   `-CXXForRangeStmt {{.*}}
// CHECK-NEXT:   |     |-<<<NULL>>>
// CHECK-NEXT:   |     |-DeclStmt {{.*}}
// CHECK-NEXT:   |     | `-VarDecl {{.*}} implicit used __range1 'const A &' cinit
// CHECK-NEXT:   |     |   |-ExprWithCleanups {{.*}} 'const A':'const P2718R0::A' lvalue
// CHECK-NEXT:   |     |   | `-CallExpr {{.*}} 'const A':'const P2718R0::A' lvalue
// CHECK-NEXT:   |     |   |   |-ImplicitCastExpr {{.*}} 'const A &(*)(const A &)' <FunctionToPointerDecay>
// CHECK-NEXT:   |     |   |   | `-DeclRefExpr {{.*}} 'const A &(const A &)' lvalue Function {{.*}} 'f1' 'const A &(const A &)'
// CHECK-NEXT:   |     |   |   `-MaterializeTemporaryExpr {{.*}} 'const A':'const P2718R0::A' lvalue extended by Var {{.*}} '__range1' 'const A &'
// CHECK-NEXT:   |     |   |     `-ImplicitCastExpr {{.*}} 'const A':'const P2718R0::A' <NoOp>
// CHECK-NEXT:   |     |   |       `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT:   |     |   |         `-CallExpr {{.*}} 'A':'P2718R0::A'
// CHECK-NEXT:   |     |   |           `-ImplicitCastExpr {{.*}} 'A (*)()' <FunctionToPointerDecay>
// CHECK-NEXT:   |     |   |             `-DeclRefExpr {{.*}} 'A ()' lvalue Function {{.*}} 'g' 'A ()'
// CHECK-NEXT:   |     |   `-typeDetails: LValueReferenceType {{.*}} 'const A &'
// CHECK-NEXT:   |     |     `-qualTypeDetail: QualType {{.*}} 'const A' const
// CHECK-NEXT:   |     |       `-typeDetails: ElaboratedType {{.*}} 'A' sugar
// CHECK-NEXT:   |     |         `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT:   |     |           `-CXXRecord {{.*}} 'A'
// CHECK-NEXT:   |     |-DeclStmt {{.*}}
// CHECK-NEXT:   |     | `-VarDecl {{.*}} implicit used __begin1 'const int *' cinit
// CHECK-NEXT:   |     |   |-CXXMemberCallExpr {{.*}} 'const int *'
// CHECK-NEXT:   |     |   | `-MemberExpr {{.*}} '<bound member function type>' .begin {{.*}}
// CHECK-NEXT:   |     |   |   `-DeclRefExpr {{.*}} 'const A':'const P2718R0::A' lvalue Var {{.*}} '__range1' 'const A &'
// CHECK-NEXT:   |     |   `-typeDetails: AutoType {{.*}} 'const int *' sugar
// CHECK-NEXT:   |     |     `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT:   |     |       `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT:   |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:   |     |-DeclStmt {{.*}}
// CHECK-NEXT:   |     | `-VarDecl {{.*}} implicit used __end1 'const int *' cinit
// CHECK-NEXT:   |     |   |-CXXMemberCallExpr {{.*}} 'const int *'
// CHECK-NEXT:   |     |   | `-MemberExpr {{.*}} '<bound member function type>' .end {{.*}}
// CHECK-NEXT:   |     |   |   `-DeclRefExpr {{.*}} 'const A':'const P2718R0::A' lvalue Var {{.*}} '__range1' 'const A &'
// CHECK-NEXT:   |     |   `-typeDetails: AutoType {{.*}} 'const int *' sugar
// CHECK-NEXT:   |     |     `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT:   |     |       `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT:   |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:   |     |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT:   |     | |-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT:   |     | | `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT:   |     | `-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT:   |     |   `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__end1' 'const int *'
// CHECK-NEXT:   |     |-UnaryOperator {{.*}} 'const int *' lvalue prefix '++'
// CHECK-NEXT:   |     | `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT:   |     |-DeclStmt {{.*}}
// CHECK-NEXT:   |     | `-VarDecl {{.*}} used e 'int' cinit
// CHECK-NEXT:   |     |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:   |     |   | `-UnaryOperator {{.*}} 'const int' lvalue prefix '*' cannot overflow
// CHECK-NEXT:   |     |   |   `-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT:   |     |   |     `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT:   |     |   `-typeDetails: AutoType {{.*}} 'int' sugar
// CHECK-NEXT:   |     |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:   |     `-CompoundAssignOperator {{.*}} 'int' lvalue '+=' ComputeLHSTy='int' ComputeResultTy='int'
// CHECK-NEXT:   |       |-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'sum' 'int'
// CHECK-NEXT:   |       `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:   |         `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'e' 'int'

struct B : A {};
int (&f(const A *))[3];
const A *g(const A &);
void bar(int) {}

// CHECK-NEXT: |-CXXRecordDecl {{.*}} referenced struct B definition
// CHECK-NEXT: | |-DefinitionData aggregate standard_layout has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK-NEXT: | | |-DefaultConstructor exists non_trivial constexpr defaulted_is_constexpr
// CHECK-NEXT: | | |-CopyConstructor simple trivial has_const_param needs_overload_resolution implicit_has_const_param
// CHECK-NEXT: | | |-MoveConstructor exists simple trivial needs_overload_resolution
// CHECK-NEXT: | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | |-MoveAssignment exists simple trivial needs_overload_resolution
// CHECK-NEXT: | | `-Destructor simple non_trivial constexpr needs_overload_resolution
// CHECK-NEXT: | |-public 'A':'P2718R0::A'
// CHECK-NEXT: | |-CXXRecordDecl {{.*}} implicit struct B
// CHECK-NEXT: | |-CXXConstructorDecl {{.*}} implicit constexpr B 'void (const B &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT: | | `-ParmVarDecl {{.*}} 'const B &'
// CHECK-NEXT: | |   `-typeDetails: LValueReferenceType {{.*}} 'const B &'
// CHECK-NEXT: | |     `-qualTypeDetail: QualType {{.*}} 'const B' const
// CHECK-NEXT: | |       `-typeDetails: ElaboratedType {{.*}} 'B' sugar
// CHECK-NEXT: | |         `-typeDetails: RecordType {{.*}} 'P2718R0::B'
// CHECK-NEXT: | |           `-CXXRecord {{.*}} 'B'
// CHECK-NEXT: | |-CXXConstructorDecl {{.*}} implicit constexpr B 'void (B &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT: | | `-ParmVarDecl {{.*}} 'B &&'
// CHECK-NEXT: | |   `-typeDetails: RValueReferenceType {{.*}} 'B &&'
// CHECK-NEXT: | |     `-typeDetails: ElaboratedType {{.*}} 'B' sugar
// CHECK-NEXT: | |       `-typeDetails: RecordType {{.*}} 'P2718R0::B'
// CHECK-NEXT: | |         `-CXXRecord {{.*}} 'B'
// CHECK-NEXT: | |-CXXMethodDecl {{.*}} implicit constexpr operator= 'B &(B &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT: | | `-ParmVarDecl {{.*}} 'B &&'
// CHECK-NEXT: | |   `-typeDetails: RValueReferenceType {{.*}} 'B &&'
// CHECK-NEXT: | |     `-typeDetails: ElaboratedType {{.*}} 'B' sugar
// CHECK-NEXT: | |       `-typeDetails: RecordType {{.*}} 'P2718R0::B'
// CHECK-NEXT: | |         `-CXXRecord {{.*}} 'B'
// CHECK-NEXT: | |-CXXDestructorDecl {{.*}} implicit used constexpr ~B 'void () noexcept' inline default
// CHECK-NEXT: | | `-CompoundStmt {{.*}}
// CHECK-NEXT: | `-CXXConstructorDecl {{.*}} implicit used constexpr B 'void () noexcept(false)' inline default
// CHECK-NEXT: |   |-CXXCtorInitializer 'A':'P2718R0::A'
// CHECK-NEXT: |   | `-CXXConstructExpr {{.*}} 'A':'P2718R0::A' 'void ()'
// CHECK-NEXT: |   `-CompoundStmt {{.*}}
// CHECK-NEXT: |-FunctionDecl {{.*}} used f 'int (&(const A *))[3]'
// CHECK-NEXT: | `-ParmVarDecl {{.*}} 'const A *'
// CHECK-NEXT: |   `-typeDetails: PointerType {{.*}} 'const A *'
// CHECK-NEXT: |     `-qualTypeDetail: QualType {{.*}} 'const A' const
// CHECK-NEXT: |       `-typeDetails: ElaboratedType {{.*}} 'A' sugar
// CHECK-NEXT: |         `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: |           `-CXXRecord {{.*}} 'A'
// CHECK-NEXT: |-FunctionDecl {{.*}} used g 'const A *(const A &)'
// CHECK-NEXT: | `-ParmVarDecl {{.*}} 'const A &'
// CHECK-NEXT: |   `-typeDetails: LValueReferenceType {{.*}} 'const A &'
// CHECK-NEXT: |     `-qualTypeDetail: QualType {{.*}} 'const A' const
// CHECK-NEXT: |       `-typeDetails: ElaboratedType {{.*}} 'A' sugar
// CHECK-NEXT: |         `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: |           `-CXXRecord {{.*}} 'A'
// CHECK-NEXT: |-FunctionDecl {{.*}} used bar 'void (int)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} 'int'
// CHECK-NEXT: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: | `-CompoundStmt {{.*}}

void test2() {
  for (auto e : f(g(B())))
    bar(e);
}

// CHECK: |-FunctionDecl {{.*}} test2 'void ()'
// CHECK-NEXT:   | `-CompoundStmt {{.*}} 
// CHECK-NEXT:   |   `-CXXForRangeStmt {{.*}} 
// CHECK-NEXT:   |     |-<<<NULL>>>
// CHECK-NEXT:   |     |-DeclStmt {{.*}} 
// CHECK-NEXT:   |     | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK-NEXT:   |     |   |-ExprWithCleanups {{.*}} 'int[3]' lvalue
// CHECK-NEXT:   |     |   | `-CallExpr {{.*}} 'int[3]' lvalue
// CHECK-NEXT:   |     |   |   |-ImplicitCastExpr {{.*}} 'int (&(*)(const A *))[3]' <FunctionToPointerDecay>
// CHECK-NEXT:   |     |   |   | `-DeclRefExpr {{.*}} 'int (&(const A *))[3]' lvalue Function {{.*}} 'f' 'int (&(const A *))[3]'
// CHECK-NEXT:   |     |   |   `-CallExpr {{.*}} 'const A *'
// CHECK-NEXT:   |     |   |     |-ImplicitCastExpr {{.*}} 'const A *(*)(const A &)' <FunctionToPointerDecay>
// CHECK-NEXT:   |     |   |     | `-DeclRefExpr {{.*}} 'const A *(const A &)' lvalue Function {{.*}} 'g' 'const A *(const A &)'
// CHECK-NEXT:   |     |   |     `-ImplicitCastExpr {{.*}} 'const A':'const P2718R0::A' lvalue <DerivedToBase (A)>
// CHECK-NEXT:   |     |   |       `-MaterializeTemporaryExpr {{.*}} 'const B':'const P2718R0::B' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT:   |     |   |         `-ImplicitCastExpr {{.*}} 'const B':'const P2718R0::B' <NoOp>
// CHECK-NEXT:   |     |   |           `-CXXBindTemporaryExpr {{.*}} 'B':'P2718R0::B' (CXXTemporary {{.*}})
// CHECK-NEXT:   |     |   |             `-CXXTemporaryObjectExpr {{.*}} 'B':'P2718R0::B' 'void () noexcept(false)' zeroing
// CHECK-NEXT:   |     |   `-typeDetails: LValueReferenceType {{.*}} 'int (&)[3]'
// CHECK-NEXT:   |     |     `-typeDetails: ParenType {{.*}} 'int[3]' sugar
// CHECK-NEXT:   |     |       `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT:   |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:   |     |-DeclStmt {{.*}} 
// CHECK-NEXT:   |     | `-VarDecl {{.*}} implicit used __begin1 'int *' cinit
// CHECK-NEXT:   |     |   |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT:   |     |   | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT:   |     |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT:   |     |     `-typeDetails: DecayedType {{.*}} 'int *' sugar
// CHECK-NEXT:   |     |       |-typeDetails: ParenType {{.*}} 'int[3]' sugar
// CHECK-NEXT:   |     |       | `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT:   |     |       |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:   |     |       `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT:   |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:   |     |-DeclStmt {{.*}} 
// CHECK-NEXT:   |     | `-VarDecl {{.*}} implicit used __end1 'int *' cinit
// CHECK-NEXT:   |     |   |-BinaryOperator {{.*}} 'int *' '+'
// CHECK-NEXT:   |     |   | |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT:   |     |   | | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK:   |     |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT:   |     |     `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT:   |     |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:   |     |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT:   |     | |-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT:   |     | | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT:   |     | `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT:   |     |   `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__end1' 'int *'
// CHECK-NEXT:   |     |-UnaryOperator {{.*}} 'int *' lvalue prefix '++'
// CHECK-NEXT:   |     | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT:   |     |-DeclStmt {{.*}} 
// CHECK-NEXT:   |     | `-VarDecl {{.*}} used e 'int' cinit
// CHECK-NEXT:   |     |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:   |     |   | `-UnaryOperator {{.*}} 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT:   |     |   |   `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT:   |     |   |     `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT:   |     |   `-typeDetails: AutoType {{.*}} 'int' sugar
// CHECK-NEXT:   |     |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:   |     `-CallExpr {{.*}} 'void'
// CHECK-NEXT:   |       |-ImplicitCastExpr {{.*}} 'void (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT:   |       | `-DeclRefExpr {{.*}} 'void (int)' lvalue Function {{.*}} 'bar' 'void (int)'
// CHECK-NEXT:   |       `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:   |         `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'e' 'int'

// Test discard statement.
struct LockGuard {
    LockGuard() {}
    ~LockGuard() {}
};

// CHECK:  |-CXXRecordDecl {{.*}} referenced struct LockGuard definition
// CHECK-NEXT:  | |-DefinitionData empty standard_layout has_user_declared_ctor can_const_default_init
// CHECK-NEXT:  | | |-DefaultConstructor exists non_trivial user_provided defaulted_is_constexpr
// CHECK-NEXT:  | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT:  | | |-MoveConstructor
// CHECK-NEXT:  | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:  | | |-MoveAssignment
// CHECK-NEXT:  | | `-Destructor non_trivial user_declared
// CHECK-NEXT:  | |-CXXRecordDecl {{.*}} implicit referenced struct LockGuard
// CHECK-NEXT:  | |-CXXConstructorDecl {{.*}} used LockGuard 'void ()' implicit-inline
// CHECK-NEXT:  | | `-CompoundStmt {{.*}} 
// CHECK-NEXT:  | |-CXXDestructorDecl {{.*}} used ~LockGuard 'void () noexcept' implicit-inline
// CHECK-NEXT:  | | `-CompoundStmt {{.*}} 
// CHECK-NEXT:  | `-CXXConstructorDecl {{.*}} implicit constexpr LockGuard 'void (const LockGuard &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:  |   `-ParmVarDecl {{.*}} 'const LockGuard &'
// CHECK-NEXT:  |     `-typeDetails: LValueReferenceType {{.*}} 'const LockGuard &'
// CHECK-NEXT:  |       `-qualTypeDetail: QualType {{.*}} 'const LockGuard' const
// CHECK-NEXT:  |         `-typeDetails: ElaboratedType {{.*}} 'LockGuard' sugar
// CHECK-NEXT:  |           `-typeDetails: RecordType {{.*}} 'P2718R0::LockGuard'
// CHECK-NEXT:  |             `-CXXRecord {{.*}} 'LockGuard'

void test3() {
  int v[] = {42, 17, 13};

  for ([[maybe_unused]] int x : static_cast<void>(LockGuard()), v)
    LockGuard guard;

  for ([[maybe_unused]] int x : (void)LockGuard(), v)
    LockGuard guard;

  for ([[maybe_unused]] int x : LockGuard(), v)
    LockGuard guard;
}

// CHECK-NEXT: |-FunctionDecl {{.*}} test3 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | `-VarDecl {{.*}} used v 'int[3]' cinit
// CHECK-NEXT: |   |   |-InitListExpr {{.*}} 'int[3]'
// CHECK-NEXT: |   |   | |-IntegerLiteral {{.*}} 'int' 42
// CHECK-NEXT: |   |   | |-IntegerLiteral {{.*}} 'int' 17
// CHECK-NEXT: |   |   | `-IntegerLiteral {{.*}} 'int' 13
// CHECK-NEXT: |   |   `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |   |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   |-CXXForRangeStmt {{.*}} 
// CHECK-NEXT: |   | |-<<<NULL>>>
// CHECK-NEXT: |   | |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK-NEXT: |   | |   |-ExprWithCleanups {{.*}} 'int[3]' lvalue
// CHECK-NEXT: |   | |   | `-BinaryOperator {{.*}} 'int[3]' lvalue ','
// CHECK-NEXT: |   | |   |   |-CXXStaticCastExpr {{.*}} 'void' static_cast<void> <ToVoid>
// CHECK-NEXT: |   | |   |   | `-MaterializeTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' xvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |   | |   |   |   `-CXXBindTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' (CXXTemporary {{.*}})
// CHECK-NEXT: |   | |   |   |     `-CXXTemporaryObjectExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK-NEXT: |   | |   |   `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} 'v' 'int[3]'
// CHECK-NEXT: |   | |   `-typeDetails: LValueReferenceType {{.*}} 'int (&)[3]'
// CHECK-NEXT: |   | |     `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |   | |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | | `-VarDecl {{.*}} implicit used __begin1 'int *' cinit
// CHECK-NEXT: |   | |   |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |   | |   | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |   | |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |   | |     `-typeDetails: DecayedType {{.*}} 'int *' sugar
// CHECK-NEXT: |   | |       |-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |   | |       | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |       `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |   | |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | | `-VarDecl {{.*}} implicit used __end1 'int *' cinit
// CHECK-NEXT: |   | |   |-BinaryOperator {{.*}} 'int *' '+'
// CHECK-NEXT: |   | |   | |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |   | |   | | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK: |   | |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |   | |     `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |   | |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT: |   | | |-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |   | | | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |   | | `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |   | |   `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__end1' 'int *'
// CHECK-NEXT: |   | |-UnaryOperator {{.*}} 'int *' lvalue prefix '++'
// CHECK-NEXT: |   | | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |   | |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | | `-VarDecl {{.*}} x 'int' cinit
// CHECK-NEXT: |   | |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |   | |   | `-UnaryOperator {{.*}} 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |   | |   |   `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |   | |   |     `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |   | |   |-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |   `-attrDetails: UnusedAttr {{.*}} maybe_unused
// CHECK-NEXT: |   | `-DeclStmt {{.*}} 
// CHECK-NEXT: |   |   `-VarDecl {{.*}} guard 'LockGuard':'P2718R0::LockGuard' callinit destroyed
// CHECK-NEXT: |   |     |-CXXConstructExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK-NEXT: |   |     `-typeDetails: ElaboratedType {{.*}} 'LockGuard' sugar
// CHECK-NEXT: |   |       `-typeDetails: RecordType {{.*}} 'P2718R0::LockGuard'
// CHECK-NEXT: |   |         `-CXXRecord {{.*}} 'LockGuard'
// CHECK-NEXT: |   |-CXXForRangeStmt {{.*}} 
// CHECK-NEXT: |   | |-<<<NULL>>>
// CHECK-NEXT: |   | |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK-NEXT: |   | |   |-ExprWithCleanups {{.*}} 'int[3]' lvalue
// CHECK-NEXT: |   | |   | `-BinaryOperator {{.*}} 'int[3]' lvalue ','
// CHECK-NEXT: |   | |   |   |-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT: |   | |   |   | `-MaterializeTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' xvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |   | |   |   |   `-CXXBindTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' (CXXTemporary {{.*}})
// CHECK-NEXT: |   | |   |   |     `-CXXTemporaryObjectExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK-NEXT: |   | |   |   `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} 'v' 'int[3]'
// CHECK-NEXT: |   | |   `-typeDetails: LValueReferenceType {{.*}} 'int (&)[3]'
// CHECK-NEXT: |   | |     `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |   | |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | | `-VarDecl {{.*}} implicit used __begin1 'int *' cinit
// CHECK-NEXT: |   | |   |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |   | |   | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |   | |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |   | |     `-typeDetails: DecayedType {{.*}} 'int *' sugar
// CHECK-NEXT: |   | |       |-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |   | |       | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |       `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |   | |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | | `-VarDecl {{.*}} implicit used __end1 'int *' cinit
// CHECK-NEXT: |   | |   |-BinaryOperator {{.*}} 'int *' '+'
// CHECK-NEXT: |   | |   | |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |   | |   | | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK: |   | |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |   | |     `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |   | |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT: |   | | |-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |   | | | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |   | | `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |   | |   `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__end1' 'int *'
// CHECK-NEXT: |   | |-UnaryOperator {{.*}} 'int *' lvalue prefix '++'
// CHECK-NEXT: |   | | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |   | |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | | `-VarDecl {{.*}} x 'int' cinit
// CHECK-NEXT: |   | |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |   | |   | `-UnaryOperator {{.*}} 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |   | |   |   `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |   | |   |     `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |   | |   |-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |   `-attrDetails: UnusedAttr {{.*}} maybe_unused
// CHECK-NEXT: |   | `-DeclStmt {{.*}} 
// CHECK-NEXT: |   |   `-VarDecl {{.*}} guard 'LockGuard':'P2718R0::LockGuard' callinit destroyed
// CHECK-NEXT: |   |     |-CXXConstructExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK-NEXT: |   |     `-typeDetails: ElaboratedType {{.*}} 'LockGuard' sugar
// CHECK-NEXT: |   |       `-typeDetails: RecordType {{.*}} 'P2718R0::LockGuard'
// CHECK-NEXT: |   |         `-CXXRecord {{.*}} 'LockGuard'
// CHECK-NEXT: |   `-CXXForRangeStmt {{.*}} 
// CHECK-NEXT: |     |-<<<NULL>>>
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK-NEXT: |     |   |-ExprWithCleanups {{.*}} 'int[3]' lvalue
// CHECK-NEXT: |     |   | `-BinaryOperator {{.*}} 'int[3]' lvalue ','
// CHECK-NEXT: |     |   |   |-MaterializeTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' xvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |   | `-CXXBindTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |   |   `-CXXTemporaryObjectExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK-NEXT: |     |   |   `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} 'v' 'int[3]'
// CHECK-NEXT: |     |   `-typeDetails: LValueReferenceType {{.*}} 'int (&)[3]'
// CHECK-NEXT: |     |     `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |     |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __begin1 'int *' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |     |   | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: DecayedType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |       |-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |     |       | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |       `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __end1 'int *' cinit
// CHECK-NEXT: |     |   |-BinaryOperator {{.*}} 'int *' '+'
// CHECK-NEXT: |     |   | |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |     |   | | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK: |     |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |     |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT: |     | |-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     | | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__end1' 'int *'
// CHECK-NEXT: |     |-UnaryOperator {{.*}} 'int *' lvalue prefix '++'
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} x 'int' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |     |   | `-UnaryOperator {{.*}} 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     |   |-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |   `-attrDetails: UnusedAttr {{.*}} maybe_unused
// CHECK-NEXT: |     `-DeclStmt {{.*}} 
// CHECK-NEXT: |       `-VarDecl {{.*}} guard 'LockGuard':'P2718R0::LockGuard' callinit destroyed
// CHECK-NEXT: |         |-CXXConstructExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK-NEXT: |         `-typeDetails: ElaboratedType {{.*}} 'LockGuard' sugar
// CHECK-NEXT: |           `-typeDetails: RecordType {{.*}} 'P2718R0::LockGuard'
// CHECK-NEXT: |             `-CXXRecord {{.*}} 'LockGuard'

// Test default arg
int (&default_arg_fn(const A & = A()))[3];

// CHECK-NEXT: |-FunctionDecl {{.*}} used default_arg_fn 'int (&(const A &))[3]'
// CHECK-NEXT: | `-ParmVarDecl {{.*}} 'const A &' cinit
// CHECK-NEXT: |   |-ExprWithCleanups {{.*}} 'const A':'const P2718R0::A' lvalue
// CHECK-NEXT: |   | `-MaterializeTemporaryExpr {{.*}} 'const A':'const P2718R0::A' lvalue
// CHECK-NEXT: |   |   `-ImplicitCastExpr {{.*}} 'const A':'const P2718R0::A' <NoOp>
// CHECK-NEXT: |   |     `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |   |       `-CXXTemporaryObjectExpr {{.*}} 'A':'P2718R0::A' 'void ()'
// CHECK-NEXT: |   `-typeDetails: LValueReferenceType {{.*}} 'const A &'
// CHECK-NEXT: |     `-qualTypeDetail: QualType {{.*}} 'const A' const
// CHECK-NEXT: |       `-typeDetails: ElaboratedType {{.*}} 'A' sugar
// CHECK-NEXT: |         `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: |           `-CXXRecord {{.*}} 'A'

void test4() {
  for (auto e : default_arg_fn()) 
    bar(e);
}

// CHECK: |-FunctionDecl {{.*}} test4 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-CXXForRangeStmt {{.*}} 
// CHECK-NEXT: |     |-<<<NULL>>>
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK-NEXT: |     |   |-ExprWithCleanups {{.*}} 'int[3]' lvalue
// CHECK-NEXT: |     |   | `-CallExpr {{.*}} 'int[3]' lvalue
// CHECK-NEXT: |     |   |   |-ImplicitCastExpr {{.*}} 'int (&(*)(const A &))[3]' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |   | `-DeclRefExpr {{.*}} 'int (&(const A &))[3]' lvalue Function {{.*}} 'default_arg_fn' 'int (&(const A &))[3]'
// CHECK-NEXT: |     |   |   `-CXXDefaultArgExpr {{.*}} <<invalid sloc>> 'const A':'const P2718R0::A' lvalue has rewritten init
// CHECK-NEXT: |     |   |     `-MaterializeTemporaryExpr {{.*}} 'const A':'const P2718R0::A' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |       `-ImplicitCastExpr {{.*}} 'const A':'const P2718R0::A' <NoOp>
// CHECK-NEXT: |     |   |         `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |           `-CXXTemporaryObjectExpr {{.*}} 'A':'P2718R0::A' 'void ()'
// CHECK-NEXT: |     |   `-typeDetails: LValueReferenceType {{.*}} 'int (&)[3]'
// CHECK-NEXT: |     |     `-typeDetails: ParenType {{.*}} 'int[3]' sugar
// CHECK-NEXT: |     |       `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __begin1 'int *' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |     |   | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: DecayedType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |       |-typeDetails: ParenType {{.*}} 'int[3]' sugar
// CHECK-NEXT: |     |       | `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |     |       |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |       `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __end1 'int *' cinit
// CHECK-NEXT: |     |   |-BinaryOperator {{.*}} 'int *' '+'
// CHECK-NEXT: |     |   | |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |     |   | | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK: |     |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |     |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT: |     | |-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     | | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__end1' 'int *'
// CHECK-NEXT: |     |-UnaryOperator {{.*}} 'int *' lvalue prefix '++'
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} used e 'int' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |     |   | `-UnaryOperator {{.*}} 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'int' sugar
// CHECK-NEXT: |     |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     `-CallExpr {{.*}} 'void'
// CHECK-NEXT: |       |-ImplicitCastExpr {{.*}} 'void (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: |       | `-DeclRefExpr {{.*}} 'void (int)' lvalue Function {{.*}} 'bar' 'void (int)'
// CHECK-NEXT: |       `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |         `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'e' 'int'

struct DefaultA {
  DefaultA() {}
  ~DefaultA() {}
};

// CHECK: |-CXXRecordDecl {{.*}} referenced struct DefaultA definition
// CHECK-NEXT: | |-DefinitionData empty standard_layout has_user_declared_ctor can_const_default_init
// CHECK-NEXT: | | |-DefaultConstructor exists non_trivial user_provided defaulted_is_constexpr
// CHECK-NEXT: | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT: | | |-MoveConstructor
// CHECK-NEXT: | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | |-MoveAssignment
// CHECK-NEXT: | | `-Destructor non_trivial user_declared
// CHECK-NEXT: | |-CXXRecordDecl {{.*}} implicit referenced struct DefaultA
// CHECK-NEXT: | |-CXXConstructorDecl {{.*}} used DefaultA 'void ()' implicit-inline
// CHECK-NEXT: | | `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |-CXXDestructorDecl {{.*}} used ~DefaultA 'void () noexcept' implicit-inline
// CHECK-NEXT: | | `-CompoundStmt {{.*}} 
// CHECK-NEXT: | `-CXXConstructorDecl {{.*}} implicit constexpr DefaultA 'void (const DefaultA &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT: |   `-ParmVarDecl {{.*}} 'const DefaultA &'
// CHECK-NEXT: |     `-typeDetails: LValueReferenceType {{.*}} 'const DefaultA &'
// CHECK-NEXT: |       `-qualTypeDetail: QualType {{.*}} 'const DefaultA' const
// CHECK-NEXT: |         `-typeDetails: ElaboratedType {{.*}} 'DefaultA' sugar
// CHECK-NEXT: |           `-typeDetails: RecordType {{.*}} 'P2718R0::DefaultA'
// CHECK-NEXT: |             `-CXXRecord {{.*}} 'DefaultA'

A foo(const A&, const DefaultA &Default = DefaultA()) {
  return A();
}

// CHECK: |-FunctionDecl {{.*}} used foo 'A (const A &, const DefaultA &)'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} 'const A &'
// CHECK-NEXT: | | `-typeDetails: LValueReferenceType {{.*}} 'const A &'
// CHECK-NEXT: | |   `-qualTypeDetail: QualType {{.*}} 'const A' const
// CHECK-NEXT: | |     `-typeDetails: ElaboratedType {{.*}} 'A' sugar
// CHECK-NEXT: | |       `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: | |         `-CXXRecord {{.*}} 'A'
// CHECK-NEXT: | |-ParmVarDecl {{.*}} Default 'const DefaultA &' cinit
// CHECK-NEXT: | | |-ExprWithCleanups {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue
// CHECK-NEXT: | | | `-MaterializeTemporaryExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue
// CHECK-NEXT: | | |   `-ImplicitCastExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK-NEXT: | | |     `-CXXBindTemporaryExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' (CXXTemporary {{.*}})
// CHECK-NEXT: | | |       `-CXXTemporaryObjectExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK-NEXT: | | `-typeDetails: LValueReferenceType {{.*}} 'const DefaultA &'
// CHECK-NEXT: | |   `-qualTypeDetail: QualType {{.*}} 'const DefaultA' const
// CHECK-NEXT: | |     `-typeDetails: ElaboratedType {{.*}} 'DefaultA' sugar
// CHECK-NEXT: | |       `-typeDetails: RecordType {{.*}} 'P2718R0::DefaultA'
// CHECK-NEXT: | |         `-CXXRecord {{.*}} 'DefaultA'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-ReturnStmt {{.*}} 
// CHECK-NEXT: |     `-ExprWithCleanups {{.*}} 'A':'P2718R0::A'
// CHECK-NEXT: |       `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |         `-CXXTemporaryObjectExpr {{.*}} 'A':'P2718R0::A' 'void ()'

void test5() {
  for (auto e : default_arg_fn(foo(foo(foo(A())))))
    bar(e);
}

// CHECK: |-FunctionDecl {{.*}} test5 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-CXXForRangeStmt {{.*}} 
// CHECK-NEXT: |     |-<<<NULL>>>
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK-NEXT: |     |   |-ExprWithCleanups {{.*}} 'int[3]' lvalue
// CHECK-NEXT: |     |   | `-CallExpr {{.*}} 'int[3]' lvalue
// CHECK-NEXT: |     |   |   |-ImplicitCastExpr {{.*}} 'int (&(*)(const A &))[3]' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |   | `-DeclRefExpr {{.*}} 'int (&(const A &))[3]' lvalue Function {{.*}} 'default_arg_fn' 'int (&(const A &))[3]'
// CHECK-NEXT: |     |   |   `-MaterializeTemporaryExpr {{.*}} 'const A':'const P2718R0::A' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |     `-ImplicitCastExpr {{.*}} 'const A':'const P2718R0::A' <NoOp>
// CHECK-NEXT: |     |   |       `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |         `-CallExpr {{.*}} 'A':'P2718R0::A'
// CHECK-NEXT: |     |   |           |-ImplicitCastExpr {{.*}} 'A (*)(const A &, const DefaultA &)' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |           | `-DeclRefExpr {{.*}} 'A (const A &, const DefaultA &)' lvalue Function {{.*}} 'foo' 'A (const A &, const DefaultA &)'
// CHECK-NEXT: |     |   |           |-MaterializeTemporaryExpr {{.*}} 'const A':'const P2718R0::A' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |           | `-ImplicitCastExpr {{.*}} 'const A':'const P2718R0::A' <NoOp>
// CHECK-NEXT: |     |   |           |   `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |           |     `-CallExpr {{.*}} 'A':'P2718R0::A'
// CHECK-NEXT: |     |   |           |       |-ImplicitCastExpr {{.*}} 'A (*)(const A &, const DefaultA &)' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |           |       | `-DeclRefExpr {{.*}} 'A (const A &, const DefaultA &)' lvalue Function {{.*}} 'foo' 'A (const A &, const DefaultA &)'
// CHECK-NEXT: |     |   |           |       |-MaterializeTemporaryExpr {{.*}} 'const A':'const P2718R0::A' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |           |       | `-ImplicitCastExpr {{.*}} 'const A':'const P2718R0::A' <NoOp>
// CHECK-NEXT: |     |   |           |       |   `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |           |       |     `-CallExpr {{.*}} 'A':'P2718R0::A'
// CHECK-NEXT: |     |   |           |       |       |-ImplicitCastExpr {{.*}} 'A (*)(const A &, const DefaultA &)' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |           |       |       | `-DeclRefExpr {{.*}} 'A (const A &, const DefaultA &)' lvalue Function {{.*}} 'foo' 'A (const A &, const DefaultA &)'
// CHECK-NEXT: |     |   |           |       |       |-MaterializeTemporaryExpr {{.*}} 'const A':'const P2718R0::A' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |           |       |       | `-ImplicitCastExpr {{.*}} 'const A':'const P2718R0::A' <NoOp>
// CHECK-NEXT: |     |   |           |       |       |   `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |           |       |       |     `-CXXTemporaryObjectExpr {{.*}} 'A':'P2718R0::A' 'void ()'
// CHECK-NEXT: |     |   |           |       |       `-CXXDefaultArgExpr {{.*}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK-NEXT: |     |   |           |       |         `-MaterializeTemporaryExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |           |       |           `-ImplicitCastExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK-NEXT: |     |   |           |       |             `-CXXBindTemporaryExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |           |       |               `-CXXTemporaryObjectExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK-NEXT: |     |   |           |       `-CXXDefaultArgExpr {{.*}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK-NEXT: |     |   |           |         `-MaterializeTemporaryExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |           |           `-ImplicitCastExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK-NEXT: |     |   |           |             `-CXXBindTemporaryExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |           |               `-CXXTemporaryObjectExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK-NEXT: |     |   |           `-CXXDefaultArgExpr {{.*}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK-NEXT: |     |   |             `-MaterializeTemporaryExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |               `-ImplicitCastExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK-NEXT: |     |   |                 `-CXXBindTemporaryExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |                   `-CXXTemporaryObjectExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK-NEXT: |     |   `-typeDetails: LValueReferenceType {{.*}} 'int (&)[3]'
// CHECK-NEXT: |     |     `-typeDetails: ParenType {{.*}} 'int[3]' sugar
// CHECK-NEXT: |     |       `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __begin1 'int *' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |     |   | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: DecayedType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |       |-typeDetails: ParenType {{.*}} 'int[3]' sugar
// CHECK-NEXT: |     |       | `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |     |       |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |       `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __end1 'int *' cinit
// CHECK-NEXT: |     |   |-BinaryOperator {{.*}} 'int *' '+'
// CHECK-NEXT: |     |   | |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |     |   | | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK: |     |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |     |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT: |     | |-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     | | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__end1' 'int *'
// CHECK-NEXT: |     |-UnaryOperator {{.*}} 'int *' lvalue prefix '++'
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} used e 'int' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |     |   | `-UnaryOperator {{.*}} 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'int' sugar
// CHECK-NEXT: |     |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     `-CallExpr {{.*}} 'void'
// CHECK-NEXT: |       |-ImplicitCastExpr {{.*}} 'void (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: |       | `-DeclRefExpr {{.*}} 'void (int)' lvalue Function {{.*}} 'bar' 'void (int)'
// CHECK-NEXT: |       `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |         `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'e' 'int'

struct C : public A {
  C() {}
  C(int, const C &, const DefaultA & = DefaultA()) {}
};

// CHECK: |-CXXRecordDecl {{.*}} referenced struct C definition
// CHECK-NEXT: | |-DefinitionData standard_layout has_user_declared_ctor can_const_default_init
// CHECK-NEXT: | | |-DefaultConstructor exists non_trivial user_provided defaulted_is_constexpr
// CHECK-NEXT: | | |-CopyConstructor simple trivial has_const_param needs_overload_resolution implicit_has_const_param
// CHECK-NEXT: | | |-MoveConstructor exists simple trivial needs_overload_resolution
// CHECK-NEXT: | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | |-MoveAssignment exists simple trivial needs_overload_resolution
// CHECK-NEXT: | | `-Destructor simple non_trivial constexpr needs_overload_resolution
// CHECK-NEXT: | |-public 'A':'P2718R0::A'
// CHECK-NEXT: | |-CXXRecordDecl {{.*}} implicit referenced struct C
// CHECK-NEXT: | |-CXXConstructorDecl {{.*}} used C 'void ()' implicit-inline
// CHECK-NEXT: | | |-CXXCtorInitializer 'A':'P2718R0::A'
// CHECK-NEXT: | | | `-CXXConstructExpr {{.*}} 'A':'P2718R0::A' 'void ()'
// CHECK-NEXT: | | `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |-CXXConstructorDecl {{.*}} used C 'void (int, const C &, const DefaultA &)' implicit-inline
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} 'int'
// CHECK-NEXT: | | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} 'const C &'
// CHECK-NEXT: | | | `-typeDetails: LValueReferenceType {{.*}} 'const C &'
// CHECK-NEXT: | | |   `-qualTypeDetail: QualType {{.*}} 'const C' const
// CHECK-NEXT: | | |     `-typeDetails: ElaboratedType {{.*}} 'C' sugar
// CHECK-NEXT: | | |       `-typeDetails: RecordType {{.*}} 'P2718R0::C'
// CHECK-NEXT: | | |         `-CXXRecord {{.*}} 'C'
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} 'const DefaultA &' cinit
// CHECK-NEXT: | | | |-ExprWithCleanups {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue
// CHECK-NEXT: | | | | `-MaterializeTemporaryExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue
// CHECK-NEXT: | | | |   `-ImplicitCastExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK-NEXT: | | | |     `-CXXBindTemporaryExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' (CXXTemporary {{.*}})
// CHECK-NEXT: | | | |       `-CXXTemporaryObjectExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK-NEXT: | | | `-typeDetails: LValueReferenceType {{.*}} 'const DefaultA &'
// CHECK-NEXT: | | |   `-qualTypeDetail: QualType {{.*}} 'const DefaultA' const
// CHECK-NEXT: | | |     `-typeDetails: ElaboratedType {{.*}} 'DefaultA' sugar
// CHECK-NEXT: | | |       `-typeDetails: RecordType {{.*}} 'P2718R0::DefaultA'
// CHECK-NEXT: | | |         `-CXXRecord {{.*}} 'DefaultA'
// CHECK-NEXT: | | |-CXXCtorInitializer 'A':'P2718R0::A'
// CHECK-NEXT: | | | `-CXXConstructExpr {{.*}} 'A':'P2718R0::A' 'void ()'
// CHECK-NEXT: | | `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |-CXXConstructorDecl {{.*}} implicit constexpr C 'void (const C &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT: | | `-ParmVarDecl {{.*}} 'const C &'
// CHECK-NEXT: | |   `-typeDetails: LValueReferenceType {{.*}} 'const C &'
// CHECK-NEXT: | |     `-qualTypeDetail: QualType {{.*}} 'const C' const
// CHECK-NEXT: | |       `-typeDetails: ElaboratedType {{.*}} 'C' sugar
// CHECK-NEXT: | |         `-typeDetails: RecordType {{.*}} 'P2718R0::C'
// CHECK-NEXT: | |           `-CXXRecord {{.*}} 'C'
// CHECK-NEXT: | |-CXXConstructorDecl {{.*}} implicit constexpr C 'void (C &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT: | | `-ParmVarDecl {{.*}} 'C &&'
// CHECK-NEXT: | |   `-typeDetails: RValueReferenceType {{.*}} 'C &&'
// CHECK-NEXT: | |     `-typeDetails: ElaboratedType {{.*}} 'C' sugar
// CHECK-NEXT: | |       `-typeDetails: RecordType {{.*}} 'P2718R0::C'
// CHECK-NEXT: | |         `-CXXRecord {{.*}} 'C'
// CHECK-NEXT: | |-CXXMethodDecl {{.*}} implicit constexpr operator= 'C &(C &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT: | | `-ParmVarDecl {{.*}} 'C &&'
// CHECK-NEXT: | |   `-typeDetails: RValueReferenceType {{.*}} 'C &&'
// CHECK-NEXT: | |     `-typeDetails: ElaboratedType {{.*}} 'C' sugar
// CHECK-NEXT: | |       `-typeDetails: RecordType {{.*}} 'P2718R0::C'
// CHECK-NEXT: | |         `-CXXRecord {{.*}} 'C'
// CHECK-NEXT: | `-CXXDestructorDecl {{.*}} implicit used constexpr ~C 'void () noexcept' inline default
// CHECK-NEXT: |   `-CompoundStmt {{.*}} 

void test6() {
  for (auto e : C(0, C(0, C(0, C()))))
    bar(e);
}

// CHECK: |-FunctionDecl {{.*}} test6 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-CXXForRangeStmt {{.*}} 
// CHECK-NEXT: |     |-<<<NULL>>>
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __range1 'C &&' cinit
// CHECK-NEXT: |     |   |-ExprWithCleanups {{.*}} 'C':'P2718R0::C' xvalue
// CHECK-NEXT: |     |   | `-MaterializeTemporaryExpr {{.*}} 'C':'P2718R0::C' xvalue extended by Var {{.*}} '__range1' 'C &&'
// CHECK-NEXT: |     |   |   `-CXXBindTemporaryExpr {{.*}} 'C':'P2718R0::C' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |     `-CXXTemporaryObjectExpr {{.*}} 'C':'P2718R0::C' 'void (int, const C &, const DefaultA &)'
// CHECK-NEXT: |     |   |       |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |     |   |       |-MaterializeTemporaryExpr {{.*}} 'const C':'const P2718R0::C' lvalue extended by Var {{.*}} '__range1' 'C &&'
// CHECK-NEXT: |     |   |       | `-ImplicitCastExpr {{.*}} 'const C':'const P2718R0::C' <NoOp>
// CHECK-NEXT: |     |   |       |   `-CXXBindTemporaryExpr {{.*}} 'C':'P2718R0::C' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |       |     `-CXXTemporaryObjectExpr {{.*}} 'C':'P2718R0::C' 'void (int, const C &, const DefaultA &)'
// CHECK-NEXT: |     |   |       |       |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |     |   |       |       |-MaterializeTemporaryExpr {{.*}} 'const C':'const P2718R0::C' lvalue extended by Var {{.*}} '__range1' 'C &&'
// CHECK-NEXT: |     |   |       |       | `-ImplicitCastExpr {{.*}} 'const C':'const P2718R0::C' <NoOp>
// CHECK-NEXT: |     |   |       |       |   `-CXXBindTemporaryExpr {{.*}} 'C':'P2718R0::C' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |       |       |     `-CXXTemporaryObjectExpr {{.*}} 'C':'P2718R0::C' 'void (int, const C &, const DefaultA &)'
// CHECK-NEXT: |     |   |       |       |       |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |     |   |       |       |       |-MaterializeTemporaryExpr {{.*}} 'const C':'const P2718R0::C' lvalue extended by Var {{.*}} '__range1' 'C &&'
// CHECK-NEXT: |     |   |       |       |       | `-ImplicitCastExpr {{.*}} 'const C':'const P2718R0::C' <NoOp>
// CHECK-NEXT: |     |   |       |       |       |   `-CXXBindTemporaryExpr {{.*}} 'C':'P2718R0::C' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |       |       |       |     `-CXXTemporaryObjectExpr {{.*}} 'C':'P2718R0::C' 'void ()'
// CHECK-NEXT: |     |   |       |       |       `-CXXDefaultArgExpr {{.*}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK-NEXT: |     |   |       |       |         `-MaterializeTemporaryExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var {{.*}} '__range1' 'C &&'
// CHECK-NEXT: |     |   |       |       |           `-ImplicitCastExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK-NEXT: |     |   |       |       |             `-CXXBindTemporaryExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |       |       |               `-CXXTemporaryObjectExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK-NEXT: |     |   |       |       `-CXXDefaultArgExpr {{.*}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK-NEXT: |     |   |       |         `-MaterializeTemporaryExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var {{.*}} '__range1' 'C &&'
// CHECK-NEXT: |     |   |       |           `-ImplicitCastExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK-NEXT: |     |   |       |             `-CXXBindTemporaryExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |       |               `-CXXTemporaryObjectExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK-NEXT: |     |   |       `-CXXDefaultArgExpr {{.*}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK-NEXT: |     |   |         `-MaterializeTemporaryExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var {{.*}} '__range1' 'C &&'
// CHECK-NEXT: |     |   |           `-ImplicitCastExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK-NEXT: |     |   |             `-CXXBindTemporaryExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |               `-CXXTemporaryObjectExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK-NEXT: |     |   `-typeDetails: RValueReferenceType {{.*}} 'C &&'
// CHECK-NEXT: |     |     `-typeDetails: AutoType {{.*}} 'C' sugar
// CHECK-NEXT: |     |       `-typeDetails: ElaboratedType {{.*}} 'C' sugar
// CHECK-NEXT: |     |         `-typeDetails: RecordType {{.*}} 'P2718R0::C'
// CHECK-NEXT: |     |           `-CXXRecord {{.*}} 'C'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __begin1 'const int *' cinit
// CHECK-NEXT: |     |   |-CXXMemberCallExpr {{.*}} 'const int *'
// CHECK-NEXT: |     |   | `-MemberExpr {{.*}} '<bound member function type>' .begin {{.*}}
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'const P2718R0::A' lvalue <UncheckedDerivedToBase (A)>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'C':'P2718R0::C' lvalue Var {{.*}} '__range1' 'C &&'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'const int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT: |     |       `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __end1 'const int *' cinit
// CHECK-NEXT: |     |   |-CXXMemberCallExpr {{.*}} 'const int *'
// CHECK-NEXT: |     |   | `-MemberExpr {{.*}} '<bound member function type>' .end {{.*}}
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'const P2718R0::A' lvalue <UncheckedDerivedToBase (A)>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'C':'P2718R0::C' lvalue Var {{.*}} '__range1' 'C &&'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'const int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT: |     |       `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT: |     | |-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT: |     | | `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__end1' 'const int *'
// CHECK-NEXT: |     |-UnaryOperator {{.*}} 'const int *' lvalue prefix '++'
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} used e 'int' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |     |   | `-UnaryOperator {{.*}} 'const int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'int' sugar
// CHECK-NEXT: |     |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     `-CallExpr {{.*}} 'void'
// CHECK-NEXT: |       |-ImplicitCastExpr {{.*}} 'void (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: |       | `-DeclRefExpr {{.*}} 'void (int)' lvalue Function {{.*}} 'bar' 'void (int)'
// CHECK-NEXT: |       `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |         `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'e' 'int'

// Test member call
void test7() {
  for (auto e : g().r().g().r().g().r().g())
    bar(e);
}

// CHECK: |-FunctionDecl {{.*}} test7 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-CXXForRangeStmt {{.*}} 
// CHECK-NEXT: |     |-<<<NULL>>>
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __range1 'A &&' cinit
// CHECK-NEXT: |     |   |-ExprWithCleanups {{.*}} 'A':'P2718R0::A' xvalue
// CHECK-NEXT: |     |   | `-MaterializeTemporaryExpr {{.*}} 'A':'P2718R0::A' xvalue extended by Var {{.*}} '__range1' 'A &&'
// CHECK-NEXT: |     |   |   `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |     `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A'
// CHECK-NEXT: |     |   |       `-MemberExpr {{.*}} '<bound member function type>' .g {{.*}}
// CHECK-NEXT: |     |   |         `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A' lvalue
// CHECK-NEXT: |     |   |           `-MemberExpr {{.*}} '<bound member function type>' .r {{.*}}
// CHECK-NEXT: |     |   |             `-MaterializeTemporaryExpr {{.*}} 'A':'P2718R0::A' xvalue extended by Var {{.*}} '__range1' 'A &&'
// CHECK-NEXT: |     |   |               `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |                 `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A'
// CHECK-NEXT: |     |   |                   `-MemberExpr {{.*}} '<bound member function type>' .g {{.*}}
// CHECK-NEXT: |     |   |                     `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A' lvalue
// CHECK-NEXT: |     |   |                       `-MemberExpr {{.*}} '<bound member function type>' .r {{.*}}
// CHECK-NEXT: |     |   |                         `-MaterializeTemporaryExpr {{.*}} 'A':'P2718R0::A' xvalue extended by Var {{.*}} '__range1' 'A &&'
// CHECK-NEXT: |     |   |                           `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |                             `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A'
// CHECK-NEXT: |     |   |                               `-MemberExpr {{.*}} '<bound member function type>' .g {{.*}}
// CHECK-NEXT: |     |   |                                 `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A' lvalue
// CHECK-NEXT: |     |   |                                   `-MemberExpr {{.*}} '<bound member function type>' .r {{.*}}
// CHECK-NEXT: |     |   |                                     `-MaterializeTemporaryExpr {{.*}} 'A':'P2718R0::A' xvalue extended by Var {{.*}} '__range1' 'A &&'
// CHECK-NEXT: |     |   |                                       `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |                                         `-CallExpr {{.*}} 'A':'P2718R0::A'
// CHECK-NEXT: |     |   |                                           `-ImplicitCastExpr {{.*}} 'A (*)()' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |                                             `-DeclRefExpr {{.*}} 'A ()' lvalue Function {{.*}} 'g' 'A ()'
// CHECK-NEXT: |     |   `-typeDetails: RValueReferenceType {{.*}} 'A &&'
// CHECK-NEXT: |     |     `-typeDetails: AutoType {{.*}} 'A' sugar
// CHECK-NEXT: |     |       `-typeDetails: ElaboratedType {{.*}} 'A' sugar
// CHECK-NEXT: |     |         `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: |     |           `-CXXRecord {{.*}} 'A'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __begin1 'const int *' cinit
// CHECK-NEXT: |     |   |-CXXMemberCallExpr {{.*}} 'const int *'
// CHECK-NEXT: |     |   | `-MemberExpr {{.*}} '<bound member function type>' .begin {{.*}}
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'const P2718R0::A' lvalue <NoOp>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'A':'P2718R0::A' lvalue Var {{.*}} '__range1' 'A &&'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'const int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT: |     |       `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __end1 'const int *' cinit
// CHECK-NEXT: |     |   |-CXXMemberCallExpr {{.*}} 'const int *'
// CHECK-NEXT: |     |   | `-MemberExpr {{.*}} '<bound member function type>' .end {{.*}}
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'const P2718R0::A' lvalue <NoOp>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'A':'P2718R0::A' lvalue Var {{.*}} '__range1' 'A &&'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'const int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT: |     |       `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT: |     | |-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT: |     | | `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__end1' 'const int *'
// CHECK-NEXT: |     |-UnaryOperator {{.*}} 'const int *' lvalue prefix '++'
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} used e 'int' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |     |   | `-UnaryOperator {{.*}} 'const int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'int' sugar
// CHECK-NEXT: |     |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     `-CallExpr {{.*}} 'void'
// CHECK-NEXT: |       |-ImplicitCastExpr {{.*}} 'void (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: |       | `-DeclRefExpr {{.*}} 'void (int)' lvalue Function {{.*}} 'bar' 'void (int)'
// CHECK-NEXT: |       `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |         `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'e' 'int'


// Test basic && dependent context
template <typename T> T dg() { return T(); }

// CHECK: |-FunctionTemplateDecl {{.*}} dg
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl {{.*}} dg 'T ()'
// CHECK-NEXT: | | `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |   `-ReturnStmt {{.*}} 
// CHECK-NEXT: | |     `-CXXUnresolvedConstructExpr {{.*}} 'T' 'T'
// CHECK-NEXT: | `-FunctionDecl {{.*}} used dg 'P2718R0::A ()' implicit_instantiation
// CHECK-NEXT: |   |-TemplateArgument type 'P2718R0::A'
// CHECK-NEXT: |   | `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: |   |   `-CXXRecord {{.*}} 'A'
// CHECK-NEXT: |   `-CompoundStmt {{.*}} 
// CHECK-NEXT: |     `-ReturnStmt {{.*}} 
// CHECK-NEXT: |       `-ExprWithCleanups {{.*}} 'P2718R0::A'
// CHECK-NEXT: |         `-CXXBindTemporaryExpr {{.*}} 'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |           `-CXXTemporaryObjectExpr {{.*}} 'P2718R0::A' 'void ()'

template <typename T> const T &df1(const T &t) { return t; }

// CHECK: |-FunctionTemplateDecl {{.*}} df1
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl {{.*}} df1 'const T &(const T &)'
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} referenced t 'const T &'
// CHECK-NEXT: | | | `-typeDetails: LValueReferenceType {{.*}} 'const T &' dependent
// CHECK-NEXT: | | |   `-qualTypeDetail: QualType {{.*}} 'const T' const
// CHECK-NEXT: | | |     `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
// CHECK-NEXT: | | |       `-TemplateTypeParm {{.*}} 'T'
// CHECK-NEXT: | | `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |   `-ReturnStmt {{.*}} 
// CHECK-NEXT: | |     `-DeclRefExpr {{.*}} 'const T' lvalue ParmVar {{.*}} 't' 'const T &'
// CHECK-NEXT: | |-FunctionDecl {{.*}} used df1 'const P2718R0::A &(const P2718R0::A &)' implicit_instantiation
// CHECK-NEXT: | | |-TemplateArgument type 'P2718R0::A'
// CHECK-NEXT: | | | `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: | | |   `-CXXRecord {{.*}} 'A'
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} used t 'const P2718R0::A &'
// CHECK-NEXT: | | | `-typeDetails: LValueReferenceType {{.*}} 'const P2718R0::A &'
// CHECK-NEXT: | | |   `-qualTypeDetail: QualType {{.*}} 'const P2718R0::A' const
// CHECK-NEXT: | | |     `-typeDetails: SubstTemplateTypeParmType {{.*}} 'P2718R0::A' sugar typename depth 0 index 0 T
// CHECK-NEXT: | | |       |-FunctionTemplate {{.*}} 'df1'
// CHECK-NEXT: | | |       `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: | | |         `-CXXRecord {{.*}} 'A'
// CHECK-NEXT: | | `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |   `-ReturnStmt {{.*}} 
// CHECK-NEXT: | |     `-DeclRefExpr {{.*}} 'const P2718R0::A' lvalue ParmVar {{.*}} 't' 'const P2718R0::A &'
// CHECK-NEXT: | `-FunctionDecl {{.*}} used df1 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' implicit_instantiation
// CHECK-NEXT: |   |-TemplateArgument type 'P2718R0::LockGuard'
// CHECK-NEXT: |   | `-typeDetails: RecordType {{.*}} 'P2718R0::LockGuard'
// CHECK-NEXT: |   |   `-CXXRecord {{.*}} 'LockGuard'
// CHECK-NEXT: |   |-ParmVarDecl {{.*}} used t 'const P2718R0::LockGuard &'
// CHECK-NEXT: |   | `-typeDetails: LValueReferenceType {{.*}} 'const P2718R0::LockGuard &'
// CHECK-NEXT: |   |   `-qualTypeDetail: QualType {{.*}} 'const P2718R0::LockGuard' const
// CHECK-NEXT: |   |     `-typeDetails: SubstTemplateTypeParmType {{.*}} 'P2718R0::LockGuard' sugar typename depth 0 index 0 T
// CHECK-NEXT: |   |       |-FunctionTemplate {{.*}} 'df1'
// CHECK-NEXT: |   |       `-typeDetails: RecordType {{.*}} 'P2718R0::LockGuard'
// CHECK-NEXT: |   |         `-CXXRecord {{.*}} 'LockGuard'
// CHECK-NEXT: |   `-CompoundStmt {{.*}} 
// CHECK-NEXT: |     `-ReturnStmt {{.*}} 
// CHECK-NEXT: |       `-DeclRefExpr {{.*}} 'const P2718R0::LockGuard' lvalue ParmVar {{.*}} 't' 'const P2718R0::LockGuard &'

void test8() {
  [[maybe_unused]] int sum = 0;
  for (auto e : df1(dg<A>()))
    sum += e;
}

// CHECK: |-FunctionDecl {{.*}} test8 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | `-VarDecl {{.*}} used sum 'int' cinit
// CHECK-NEXT: |   |   |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: |   |   |-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   |   `-attrDetails: UnusedAttr {{.*}} maybe_unused
// CHECK-NEXT: |   `-CXXForRangeStmt {{.*}} 
// CHECK-NEXT: |     |-<<<NULL>>>
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __range1 'const P2718R0::A &' cinit
// CHECK-NEXT: |     |   |-ExprWithCleanups {{.*}} 'const P2718R0::A' lvalue
// CHECK-NEXT: |     |   | `-CallExpr {{.*}} 'const P2718R0::A' lvalue
// CHECK-NEXT: |     |   |   |-ImplicitCastExpr {{.*}} 'const P2718R0::A &(*)(const P2718R0::A &)' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |   | `-DeclRefExpr {{.*}} 'const P2718R0::A &(const P2718R0::A &)' lvalue Function {{.*}} 'df1' 'const P2718R0::A &(const P2718R0::A &)' (FunctionTemplate {{.*}} 'df1')
// CHECK-NEXT: |     |   |   `-MaterializeTemporaryExpr {{.*}} 'const P2718R0::A' lvalue extended by Var {{.*}} '__range1' 'const P2718R0::A &'
// CHECK-NEXT: |     |   |     `-ImplicitCastExpr {{.*}} 'const P2718R0::A' <NoOp>
// CHECK-NEXT: |     |   |       `-CXXBindTemporaryExpr {{.*}} 'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |         `-CallExpr {{.*}} 'P2718R0::A'
// CHECK-NEXT: |     |   |           `-ImplicitCastExpr {{.*}} 'P2718R0::A (*)()' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |             `-DeclRefExpr {{.*}} 'P2718R0::A ()' lvalue Function {{.*}} 'dg' 'P2718R0::A ()' (FunctionTemplate {{.*}} 'dg')
// CHECK-NEXT: |     |   `-typeDetails: LValueReferenceType {{.*}} 'const P2718R0::A &'
// CHECK-NEXT: |     |     `-qualTypeDetail: QualType {{.*}} 'const P2718R0::A' const
// CHECK-NEXT: |     |       `-typeDetails: SubstTemplateTypeParmType {{.*}} 'P2718R0::A' sugar typename depth 0 index 0 T
// CHECK-NEXT: |     |         |-FunctionTemplate {{.*}} 'df1'
// CHECK-NEXT: |     |         `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: |     |           `-CXXRecord {{.*}} 'A'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __begin1 'const int *' cinit
// CHECK-NEXT: |     |   |-CXXMemberCallExpr {{.*}} 'const int *'
// CHECK-NEXT: |     |   | `-MemberExpr {{.*}} '<bound member function type>' .begin {{.*}}
// CHECK-NEXT: |     |   |   `-DeclRefExpr {{.*}} 'const P2718R0::A' lvalue Var {{.*}} '__range1' 'const P2718R0::A &'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'const int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT: |     |       `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __end1 'const int *' cinit
// CHECK-NEXT: |     |   |-CXXMemberCallExpr {{.*}} 'const int *'
// CHECK-NEXT: |     |   | `-MemberExpr {{.*}} '<bound member function type>' .end {{.*}}
// CHECK-NEXT: |     |   |   `-DeclRefExpr {{.*}} 'const P2718R0::A' lvalue Var {{.*}} '__range1' 'const P2718R0::A &'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'const int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT: |     |       `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT: |     | |-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT: |     | | `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__end1' 'const int *'
// CHECK-NEXT: |     |-UnaryOperator {{.*}} 'const int *' lvalue prefix '++'
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} used e 'int' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |     |   | `-UnaryOperator {{.*}} 'const int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'int' sugar
// CHECK-NEXT: |     |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     `-CompoundAssignOperator {{.*}} 'int' lvalue '+=' ComputeLHSTy='int' ComputeResultTy='int'
// CHECK-NEXT: |       |-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'sum' 'int'
// CHECK-NEXT: |       `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |         `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'e' 'int'

template <typename T> int (&df2(const T *))[3];
const A *dg2(const A &);


// CHECK: |-FunctionTemplateDecl {{.*}} df2
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl {{.*}} df2 'int (&(const T *))[3]'
// CHECK-NEXT: | | `-ParmVarDecl {{.*}} 'const T *'
// CHECK-NEXT: | |   `-typeDetails: PointerType {{.*}} 'const T *' dependent
// CHECK-NEXT: | |     `-qualTypeDetail: QualType {{.*}} 'const T' const
// CHECK-NEXT: | |       `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
// CHECK-NEXT: | |         `-TemplateTypeParm {{.*}} 'T'
// CHECK-NEXT: | `-FunctionDecl {{.*}} used df2 'int (&(const P2718R0::A *))[3]' implicit_instantiation
// CHECK-NEXT: |   |-TemplateArgument type 'P2718R0::A'
// CHECK-NEXT: |   | `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: |   |   `-CXXRecord {{.*}} 'A'
// CHECK-NEXT: |   `-ParmVarDecl {{.*}} 'const P2718R0::A *'
// CHECK-NEXT: |     `-typeDetails: PointerType {{.*}} 'const P2718R0::A *'
// CHECK-NEXT: |       `-qualTypeDetail: QualType {{.*}} 'const P2718R0::A' const
// CHECK-NEXT: |         `-typeDetails: SubstTemplateTypeParmType {{.*}} 'P2718R0::A' sugar typename depth 0 index 0 T
// CHECK-NEXT: |           |-FunctionTemplate {{.*}} 'df2'
// CHECK-NEXT: |           `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: |             `-CXXRecord {{.*}} 'A'
// CHECK-NEXT: |-FunctionDecl {{.*}} used dg2 'const A *(const A &)'
// CHECK-NEXT: | `-ParmVarDecl {{.*}} 'const A &'
// CHECK-NEXT: |   `-typeDetails: LValueReferenceType {{.*}} 'const A &'
// CHECK-NEXT: |     `-qualTypeDetail: QualType {{.*}} 'const A' const
// CHECK-NEXT: |       `-typeDetails: ElaboratedType {{.*}} 'A' sugar
// CHECK-NEXT: |         `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: |           `-CXXRecord {{.*}} 'A'

void test9() {
  for (auto e : df2(dg2(B())))
    bar(e);
}

// CHECK: |-FunctionDecl {{.*}} test9 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-CXXForRangeStmt {{.*}} 
// CHECK-NEXT: |     |-<<<NULL>>>
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK-NEXT: |     |   |-ExprWithCleanups {{.*}} 'int[3]' lvalue
// CHECK-NEXT: |     |   | `-CallExpr {{.*}} 'int[3]' lvalue
// CHECK-NEXT: |     |   |   |-ImplicitCastExpr {{.*}} 'int (&(*)(const P2718R0::A *))[3]' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |   | `-DeclRefExpr {{.*}} 'int (&(const P2718R0::A *))[3]' lvalue Function {{.*}} 'df2' 'int (&(const P2718R0::A *))[3]' (FunctionTemplate {{.*}} 'df2')
// CHECK-NEXT: |     |   |   `-CallExpr {{.*}} 'const A *'
// CHECK-NEXT: |     |   |     |-ImplicitCastExpr {{.*}} 'const A *(*)(const A &)' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |     | `-DeclRefExpr {{.*}} 'const A *(const A &)' lvalue Function {{.*}} 'dg2' 'const A *(const A &)'
// CHECK-NEXT: |     |   |     `-ImplicitCastExpr {{.*}} 'const A':'const P2718R0::A' lvalue <DerivedToBase (A)>
// CHECK-NEXT: |     |   |       `-MaterializeTemporaryExpr {{.*}} 'const B':'const P2718R0::B' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |         `-ImplicitCastExpr {{.*}} 'const B':'const P2718R0::B' <NoOp>
// CHECK-NEXT: |     |   |           `-CXXBindTemporaryExpr {{.*}} 'B':'P2718R0::B' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |             `-CXXTemporaryObjectExpr {{.*}} 'B':'P2718R0::B' 'void () noexcept(false)' zeroing
// CHECK-NEXT: |     |   `-typeDetails: LValueReferenceType {{.*}} 'int (&)[3]'
// CHECK-NEXT: |     |     `-typeDetails: ParenType {{.*}} 'int[3]' sugar
// CHECK-NEXT: |     |       `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __begin1 'int *' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |     |   | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: DecayedType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |       |-typeDetails: ParenType {{.*}} 'int[3]' sugar
// CHECK-NEXT: |     |       | `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |     |       |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |       `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __end1 'int *' cinit
// CHECK-NEXT: |     |   |-BinaryOperator {{.*}} 'int *' '+'
// CHECK-NEXT: |     |   | |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |     |   | | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK: |     |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |     |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT: |     | |-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     | | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__end1' 'int *'
// CHECK-NEXT: |     |-UnaryOperator {{.*}} 'int *' lvalue prefix '++'
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} used e 'int' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |     |   | `-UnaryOperator {{.*}} 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'int' sugar
// CHECK-NEXT: |     |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     `-CallExpr {{.*}} 'void'
// CHECK-NEXT: |       |-ImplicitCastExpr {{.*}} 'void (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: |       | `-DeclRefExpr {{.*}} 'void (int)' lvalue Function {{.*}} 'bar' 'void (int)'
// CHECK-NEXT: |       `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |         `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'e' 'int'

// Test discard statement && dependent context
void test10() {
  int v[] = {42, 17, 13};

  for ([[maybe_unused]] int x : static_cast<void>(df1(LockGuard())), v)
    LockGuard guard;

  for ([[maybe_unused]] int x : (void)df1(LockGuard()), v)
    LockGuard guard;

  for ([[maybe_unused]] int x : df1(LockGuard()), df1(LockGuard()), v)
    LockGuard guard;
}

// CHECK: |-FunctionDecl {{.*}} test10 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | `-VarDecl {{.*}} used v 'int[3]' cinit
// CHECK-NEXT: |   |   |-InitListExpr {{.*}} 'int[3]'
// CHECK-NEXT: |   |   | |-IntegerLiteral {{.*}} 'int' 42
// CHECK-NEXT: |   |   | |-IntegerLiteral {{.*}} 'int' 17
// CHECK-NEXT: |   |   | `-IntegerLiteral {{.*}} 'int' 13
// CHECK-NEXT: |   |   `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |   |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   |-CXXForRangeStmt {{.*}} 
// CHECK-NEXT: |   | |-<<<NULL>>>
// CHECK-NEXT: |   | |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK-NEXT: |   | |   |-ExprWithCleanups {{.*}} 'int[3]' lvalue
// CHECK-NEXT: |   | |   | `-BinaryOperator {{.*}} 'int[3]' lvalue ','
// CHECK-NEXT: |   | |   |   |-CXXStaticCastExpr {{.*}} 'void' static_cast<void> <ToVoid>
// CHECK-NEXT: |   | |   |   | `-CallExpr {{.*}} 'const P2718R0::LockGuard' lvalue
// CHECK-NEXT: |   | |   |   |   |-ImplicitCastExpr {{.*}} 'const P2718R0::LockGuard &(*)(const P2718R0::LockGuard &)' <FunctionToPointerDecay>
// CHECK-NEXT: |   | |   |   |   | `-DeclRefExpr {{.*}} 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' lvalue Function {{.*}} 'df1' 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' (FunctionTemplate {{.*}} 'df1')
// CHECK-NEXT: |   | |   |   |   `-MaterializeTemporaryExpr {{.*}} 'const LockGuard':'const P2718R0::LockGuard' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |   | |   |   |     `-ImplicitCastExpr {{.*}} 'const LockGuard':'const P2718R0::LockGuard' <NoOp>
// CHECK-NEXT: |   | |   |   |       `-CXXBindTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' (CXXTemporary {{.*}})
// CHECK-NEXT: |   | |   |   |         `-CXXTemporaryObjectExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK-NEXT: |   | |   |   `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} 'v' 'int[3]'
// CHECK-NEXT: |   | |   `-typeDetails: LValueReferenceType {{.*}} 'int (&)[3]'
// CHECK-NEXT: |   | |     `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |   | |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | | `-VarDecl {{.*}} implicit used __begin1 'int *' cinit
// CHECK-NEXT: |   | |   |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |   | |   | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |   | |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |   | |     `-typeDetails: DecayedType {{.*}} 'int *' sugar
// CHECK-NEXT: |   | |       |-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |   | |       | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |       `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |   | |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | | `-VarDecl {{.*}} implicit used __end1 'int *' cinit
// CHECK-NEXT: |   | |   |-BinaryOperator {{.*}} 'int *' '+'
// CHECK-NEXT: |   | |   | |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |   | |   | | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK: |   | |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |   | |     `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |   | |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT: |   | | |-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |   | | | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |   | | `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |   | |   `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__end1' 'int *'
// CHECK-NEXT: |   | |-UnaryOperator {{.*}} 'int *' lvalue prefix '++'
// CHECK-NEXT: |   | | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |   | |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | | `-VarDecl {{.*}} x 'int' cinit
// CHECK-NEXT: |   | |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |   | |   | `-UnaryOperator {{.*}} 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |   | |   |   `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |   | |   |     `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |   | |   |-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |   `-attrDetails: UnusedAttr {{.*}} maybe_unused
// CHECK-NEXT: |   | `-DeclStmt {{.*}} 
// CHECK-NEXT: |   |   `-VarDecl {{.*}} guard 'LockGuard':'P2718R0::LockGuard' callinit destroyed
// CHECK-NEXT: |   |     |-CXXConstructExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK-NEXT: |   |     `-typeDetails: ElaboratedType {{.*}} 'LockGuard' sugar
// CHECK-NEXT: |   |       `-typeDetails: RecordType {{.*}} 'P2718R0::LockGuard'
// CHECK-NEXT: |   |         `-CXXRecord {{.*}} 'LockGuard'
// CHECK-NEXT: |   |-CXXForRangeStmt {{.*}} 
// CHECK-NEXT: |   | |-<<<NULL>>>
// CHECK-NEXT: |   | |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK-NEXT: |   | |   |-ExprWithCleanups {{.*}} 'int[3]' lvalue
// CHECK-NEXT: |   | |   | `-BinaryOperator {{.*}} 'int[3]' lvalue ','
// CHECK-NEXT: |   | |   |   |-CStyleCastExpr {{.*}} 'void' <ToVoid>
// CHECK-NEXT: |   | |   |   | `-CallExpr {{.*}} 'const P2718R0::LockGuard' lvalue
// CHECK-NEXT: |   | |   |   |   |-ImplicitCastExpr {{.*}} 'const P2718R0::LockGuard &(*)(const P2718R0::LockGuard &)' <FunctionToPointerDecay>
// CHECK-NEXT: |   | |   |   |   | `-DeclRefExpr {{.*}} 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' lvalue Function {{.*}} 'df1' 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' (FunctionTemplate {{.*}} 'df1')
// CHECK-NEXT: |   | |   |   |   `-MaterializeTemporaryExpr {{.*}} 'const LockGuard':'const P2718R0::LockGuard' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |   | |   |   |     `-ImplicitCastExpr {{.*}} 'const LockGuard':'const P2718R0::LockGuard' <NoOp>
// CHECK-NEXT: |   | |   |   |       `-CXXBindTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' (CXXTemporary {{.*}})
// CHECK-NEXT: |   | |   |   |         `-CXXTemporaryObjectExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK-NEXT: |   | |   |   `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} 'v' 'int[3]'
// CHECK-NEXT: |   | |   `-typeDetails: LValueReferenceType {{.*}} 'int (&)[3]'
// CHECK-NEXT: |   | |     `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |   | |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | | `-VarDecl {{.*}} implicit used __begin1 'int *' cinit
// CHECK-NEXT: |   | |   |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |   | |   | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |   | |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |   | |     `-typeDetails: DecayedType {{.*}} 'int *' sugar
// CHECK-NEXT: |   | |       |-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |   | |       | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |       `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |   | |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | | `-VarDecl {{.*}} implicit used __end1 'int *' cinit
// CHECK-NEXT: |   | |   |-BinaryOperator {{.*}} 'int *' '+'
// CHECK-NEXT: |   | |   | |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |   | |   | | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK: |   | |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |   | |     `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |   | |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT: |   | | |-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |   | | | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |   | | `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |   | |   `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__end1' 'int *'
// CHECK-NEXT: |   | |-UnaryOperator {{.*}} 'int *' lvalue prefix '++'
// CHECK-NEXT: |   | | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |   | |-DeclStmt {{.*}} 
// CHECK-NEXT: |   | | `-VarDecl {{.*}} x 'int' cinit
// CHECK-NEXT: |   | |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |   | |   | `-UnaryOperator {{.*}} 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |   | |   |   `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |   | |   |     `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |   | |   |-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |   | |   `-attrDetails: UnusedAttr {{.*}} maybe_unused
// CHECK-NEXT: |   | `-DeclStmt {{.*}} 
// CHECK-NEXT: |   |   `-VarDecl {{.*}} guard 'LockGuard':'P2718R0::LockGuard' callinit destroyed
// CHECK-NEXT: |   |     |-CXXConstructExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK-NEXT: |   |     `-typeDetails: ElaboratedType {{.*}} 'LockGuard' sugar
// CHECK-NEXT: |   |       `-typeDetails: RecordType {{.*}} 'P2718R0::LockGuard'
// CHECK-NEXT: |   |         `-CXXRecord {{.*}} 'LockGuard'
// CHECK-NEXT: |   `-CXXForRangeStmt {{.*}} 
// CHECK-NEXT: |     |-<<<NULL>>>
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK-NEXT: |     |   |-ExprWithCleanups {{.*}} 'int[3]' lvalue
// CHECK-NEXT: |     |   | `-BinaryOperator {{.*}} 'int[3]' lvalue ','
// CHECK-NEXT: |     |   |   |-BinaryOperator {{.*}} 'const P2718R0::LockGuard' lvalue ','
// CHECK-NEXT: |     |   |   | |-CallExpr {{.*}} 'const P2718R0::LockGuard' lvalue
// CHECK-NEXT: |     |   |   | | |-ImplicitCastExpr {{.*}} 'const P2718R0::LockGuard &(*)(const P2718R0::LockGuard &)' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |   | | | `-DeclRefExpr {{.*}} 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' lvalue Function {{.*}} 'df1' 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' (FunctionTemplate {{.*}} 'df1')
// CHECK-NEXT: |     |   |   | | `-MaterializeTemporaryExpr {{.*}} 'const LockGuard':'const P2718R0::LockGuard' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |   | |   `-ImplicitCastExpr {{.*}} 'const LockGuard':'const P2718R0::LockGuard' <NoOp>
// CHECK-NEXT: |     |   |   | |     `-CXXBindTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |   | |       `-CXXTemporaryObjectExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK-NEXT: |     |   |   | `-CallExpr {{.*}} 'const P2718R0::LockGuard' lvalue
// CHECK-NEXT: |     |   |   |   |-ImplicitCastExpr {{.*}} 'const P2718R0::LockGuard &(*)(const P2718R0::LockGuard &)' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |   |   | `-DeclRefExpr {{.*}} 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' lvalue Function {{.*}} 'df1' 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' (FunctionTemplate {{.*}} 'df1')
// CHECK-NEXT: |     |   |   |   `-MaterializeTemporaryExpr {{.*}} 'const LockGuard':'const P2718R0::LockGuard' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |   |     `-ImplicitCastExpr {{.*}} 'const LockGuard':'const P2718R0::LockGuard' <NoOp>
// CHECK-NEXT: |     |   |   |       `-CXXBindTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |   |         `-CXXTemporaryObjectExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK-NEXT: |     |   |   `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} 'v' 'int[3]'
// CHECK-NEXT: |     |   `-typeDetails: LValueReferenceType {{.*}} 'int (&)[3]'
// CHECK-NEXT: |     |     `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |     |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __begin1 'int *' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |     |   | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: DecayedType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |       |-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |     |       | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |       `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __end1 'int *' cinit
// CHECK-NEXT: |     |   |-BinaryOperator {{.*}} 'int *' '+'
// CHECK-NEXT: |     |   | |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |     |   | | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK: |     |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |     |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT: |     | |-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     | | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__end1' 'int *'
// CHECK-NEXT: |     |-UnaryOperator {{.*}} 'int *' lvalue prefix '++'
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} x 'int' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |     |   | `-UnaryOperator {{.*}} 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     |   |-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |   `-attrDetails: UnusedAttr {{.*}} maybe_unused
// CHECK-NEXT: |     `-DeclStmt {{.*}} 
// CHECK-NEXT: |       `-VarDecl {{.*}} guard 'LockGuard':'P2718R0::LockGuard' callinit destroyed
// CHECK-NEXT: |         |-CXXConstructExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK-NEXT: |         `-typeDetails: ElaboratedType {{.*}} 'LockGuard' sugar
// CHECK-NEXT: |           `-typeDetails: RecordType {{.*}} 'P2718R0::LockGuard'
// CHECK-NEXT: |             `-CXXRecord {{.*}} 'LockGuard'

// Test default argument && dependent context
template <typename T> int (&default_arg_fn2(const T & = T()))[3];

// CHECK: |-FunctionTemplateDecl {{.*}} default_arg_fn2
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl {{.*}} default_arg_fn2 'int (&(const T &))[3]'
// CHECK-NEXT: | | `-ParmVarDecl {{.*}} 'const T &' cinit
// CHECK-NEXT: | |   |-CXXUnresolvedConstructExpr {{.*}} 'T' 'T'
// CHECK-NEXT: | |   `-typeDetails: LValueReferenceType {{.*}} 'const T &' dependent
// CHECK-NEXT: | |     `-qualTypeDetail: QualType {{.*}} 'const T' const
// CHECK-NEXT: | |       `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
// CHECK-NEXT: | |         `-TemplateTypeParm {{.*}} 'T'
// CHECK-NEXT: | `-FunctionDecl {{.*}} used default_arg_fn2 'int (&(const P2718R0::A &))[3]' implicit_instantiation
// CHECK-NEXT: |   |-TemplateArgument type 'P2718R0::A'
// CHECK-NEXT: |   | `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: |   |   `-CXXRecord {{.*}} 'A'
// CHECK-NEXT: |   `-ParmVarDecl {{.*}} 'const P2718R0::A &' cinit
// CHECK-NEXT: |     |-ExprWithCleanups {{.*}} 'const P2718R0::A' lvalue
// CHECK-NEXT: |     | `-MaterializeTemporaryExpr {{.*}} 'const P2718R0::A' lvalue
// CHECK-NEXT: |     |   `-ImplicitCastExpr {{.*}} 'const P2718R0::A' <NoOp>
// CHECK-NEXT: |     |     `-CXXBindTemporaryExpr {{.*}} 'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |       `-CXXTemporaryObjectExpr {{.*}} 'P2718R0::A' 'void ()'
// CHECK-NEXT: |     `-typeDetails: LValueReferenceType {{.*}} 'const P2718R0::A &'
// CHECK-NEXT: |       `-qualTypeDetail: QualType {{.*}} 'const P2718R0::A' const
// CHECK-NEXT: |         `-typeDetails: SubstTemplateTypeParmType {{.*}} 'P2718R0::A' sugar typename depth 0 index 0 T
// CHECK-NEXT: |           |-FunctionTemplate {{.*}} 'default_arg_fn2'
// CHECK-NEXT: |           `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: |             `-CXXRecord {{.*}} 'A'

void test11() {
  for (auto e : default_arg_fn2<A>()) 
    bar(e);
}

// CHECK: |-FunctionDecl {{.*}} test11 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-CXXForRangeStmt {{.*}} 
// CHECK-NEXT: |     |-<<<NULL>>>
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK-NEXT: |     |   |-ExprWithCleanups {{.*}} 'int[3]' lvalue
// CHECK-NEXT: |     |   | `-CallExpr {{.*}} 'int[3]' lvalue
// CHECK-NEXT: |     |   |   |-ImplicitCastExpr {{.*}} 'int (&(*)(const P2718R0::A &))[3]' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |   | `-DeclRefExpr {{.*}} 'int (&(const P2718R0::A &))[3]' lvalue Function {{.*}} 'default_arg_fn2' 'int (&(const P2718R0::A &))[3]' (FunctionTemplate {{.*}} 'default_arg_fn2')
// CHECK-NEXT: |     |   |   `-CXXDefaultArgExpr {{.*}} <<invalid sloc>> 'const P2718R0::A' lvalue has rewritten init
// CHECK-NEXT: |     |   |     `-MaterializeTemporaryExpr {{.*}} 'const P2718R0::A' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |       `-ImplicitCastExpr {{.*}} 'const P2718R0::A' <NoOp>
// CHECK-NEXT: |     |   |         `-CXXBindTemporaryExpr {{.*}} 'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |           `-CXXTemporaryObjectExpr {{.*}} 'P2718R0::A' 'void ()'
// CHECK-NEXT: |     |   `-typeDetails: LValueReferenceType {{.*}} 'int (&)[3]'
// CHECK-NEXT: |     |     `-typeDetails: ParenType {{.*}} 'int[3]' sugar
// CHECK-NEXT: |     |       `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __begin1 'int *' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |     |   | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: DecayedType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |       |-typeDetails: ParenType {{.*}} 'int[3]' sugar
// CHECK-NEXT: |     |       | `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |     |       |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |       `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __end1 'int *' cinit
// CHECK-NEXT: |     |   |-BinaryOperator {{.*}} 'int *' '+'
// CHECK-NEXT: |     |   | |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |     |   | | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK: |     |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |     |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT: |     | |-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     | | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__end1' 'int *'
// CHECK-NEXT: |     |-UnaryOperator {{.*}} 'int *' lvalue prefix '++'
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} used e 'int' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |     |   | `-UnaryOperator {{.*}} 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'int' sugar
// CHECK-NEXT: |     |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     `-CallExpr {{.*}} 'void'
// CHECK-NEXT: |       |-ImplicitCastExpr {{.*}} 'void (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: |       | `-DeclRefExpr {{.*}} 'void (int)' lvalue Function {{.*}} 'bar' 'void (int)'
// CHECK-NEXT: |       `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |         `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'e' 'int'

template <typename T> A foo2(const T&, const DefaultA &Default = DefaultA());

// CHECK: |-FunctionTemplateDecl {{.*}} foo2
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
// CHECK-NEXT: | |-FunctionDecl {{.*}} foo2 'A (const T &, const DefaultA &)'
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} 'const T &'
// CHECK-NEXT: | | | `-typeDetails: LValueReferenceType {{.*}} 'const T &' dependent
// CHECK-NEXT: | | |   `-qualTypeDetail: QualType {{.*}} 'const T' const
// CHECK-NEXT: | | |     `-typeDetails: TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
// CHECK-NEXT: | | |       `-TemplateTypeParm {{.*}} 'T'
// CHECK-NEXT: | | `-ParmVarDecl {{.*}} Default 'const DefaultA &' cinit
// CHECK-NEXT: | |   |-ExprWithCleanups {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue
// CHECK-NEXT: | |   | `-MaterializeTemporaryExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue
// CHECK-NEXT: | |   |   `-ImplicitCastExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK-NEXT: | |   |     `-CXXBindTemporaryExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' (CXXTemporary {{.*}})
// CHECK-NEXT: | |   |       `-CXXTemporaryObjectExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK-NEXT: | |   `-typeDetails: LValueReferenceType {{.*}} 'const DefaultA &'
// CHECK-NEXT: | |     `-qualTypeDetail: QualType {{.*}} 'const DefaultA' const
// CHECK-NEXT: | |       `-typeDetails: ElaboratedType {{.*}} 'DefaultA' sugar
// CHECK-NEXT: | |         `-typeDetails: RecordType {{.*}} 'P2718R0::DefaultA'
// CHECK-NEXT: | |           `-CXXRecord {{.*}} 'DefaultA'
// CHECK-NEXT: | `-FunctionDecl {{.*}} used foo2 'A (const P2718R0::A &, const DefaultA &)' implicit_instantiation
// CHECK-NEXT: |   |-TemplateArgument type 'P2718R0::A'
// CHECK-NEXT: |   | `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: |   |   `-CXXRecord {{.*}} 'A'
// CHECK-NEXT: |   |-ParmVarDecl {{.*}} 'const P2718R0::A &'
// CHECK-NEXT: |   | `-typeDetails: LValueReferenceType {{.*}} 'const P2718R0::A &'
// CHECK-NEXT: |   |   `-qualTypeDetail: QualType {{.*}} 'const P2718R0::A' const
// CHECK-NEXT: |   |     `-typeDetails: SubstTemplateTypeParmType {{.*}} 'P2718R0::A' sugar typename depth 0 index 0 T
// CHECK-NEXT: |   |       |-FunctionTemplate {{.*}} 'foo2'
// CHECK-NEXT: |   |       `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: |   |         `-CXXRecord {{.*}} 'A'
// CHECK-NEXT: |   `-ParmVarDecl {{.*}} Default 'const DefaultA &' cinit
// CHECK-NEXT: |     |-ExprWithCleanups {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue
// CHECK-NEXT: |     | `-MaterializeTemporaryExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue
// CHECK-NEXT: |     |   `-ImplicitCastExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK-NEXT: |     |     `-CXXBindTemporaryExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |       `-CXXTemporaryObjectExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK-NEXT: |     `-typeDetails: LValueReferenceType {{.*}} 'const DefaultA &'
// CHECK-NEXT: |       `-qualTypeDetail: QualType {{.*}} 'const DefaultA' const
// CHECK-NEXT: |         `-typeDetails: ElaboratedType {{.*}} 'DefaultA' sugar
// CHECK-NEXT: |           `-typeDetails: RecordType {{.*}} 'P2718R0::DefaultA'
// CHECK-NEXT: |             `-CXXRecord {{.*}} 'DefaultA'

void test12() {
  for (auto e : default_arg_fn2(foo2(foo2(foo2(A())))))
    bar(e);
}

// CHECK: |-FunctionDecl {{.*}} test12 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-CXXForRangeStmt {{.*}} 
// CHECK-NEXT: |     |-<<<NULL>>>
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK-NEXT: |     |   |-ExprWithCleanups {{.*}} 'int[3]' lvalue
// CHECK-NEXT: |     |   | `-CallExpr {{.*}} 'int[3]' lvalue
// CHECK-NEXT: |     |   |   |-ImplicitCastExpr {{.*}} 'int (&(*)(const P2718R0::A &))[3]' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |   | `-DeclRefExpr {{.*}} 'int (&(const P2718R0::A &))[3]' lvalue Function {{.*}} 'default_arg_fn2' 'int (&(const P2718R0::A &))[3]' (FunctionTemplate {{.*}} 'default_arg_fn2')
// CHECK-NEXT: |     |   |   `-MaterializeTemporaryExpr {{.*}} 'const A':'const P2718R0::A' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |     `-ImplicitCastExpr {{.*}} 'const A':'const P2718R0::A' <NoOp>
// CHECK-NEXT: |     |   |       `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |         `-CallExpr {{.*}} 'A':'P2718R0::A'
// CHECK-NEXT: |     |   |           |-ImplicitCastExpr {{.*}} 'A (*)(const P2718R0::A &, const DefaultA &)' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |           | `-DeclRefExpr {{.*}} 'A (const P2718R0::A &, const DefaultA &)' lvalue Function {{.*}} 'foo2' 'A (const P2718R0::A &, const DefaultA &)' (FunctionTemplate {{.*}} 'foo2')
// CHECK-NEXT: |     |   |           |-MaterializeTemporaryExpr {{.*}} 'const A':'const P2718R0::A' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |           | `-ImplicitCastExpr {{.*}} 'const A':'const P2718R0::A' <NoOp>
// CHECK-NEXT: |     |   |           |   `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |           |     `-CallExpr {{.*}} 'A':'P2718R0::A'
// CHECK-NEXT: |     |   |           |       |-ImplicitCastExpr {{.*}} 'A (*)(const P2718R0::A &, const DefaultA &)' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |           |       | `-DeclRefExpr {{.*}} 'A (const P2718R0::A &, const DefaultA &)' lvalue Function {{.*}} 'foo2' 'A (const P2718R0::A &, const DefaultA &)' (FunctionTemplate {{.*}} 'foo2')
// CHECK-NEXT: |     |   |           |       |-MaterializeTemporaryExpr {{.*}} 'const A':'const P2718R0::A' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |           |       | `-ImplicitCastExpr {{.*}} 'const A':'const P2718R0::A' <NoOp>
// CHECK-NEXT: |     |   |           |       |   `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |           |       |     `-CallExpr {{.*}} 'A':'P2718R0::A'
// CHECK-NEXT: |     |   |           |       |       |-ImplicitCastExpr {{.*}} 'A (*)(const P2718R0::A &, const DefaultA &)' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |           |       |       | `-DeclRefExpr {{.*}} 'A (const P2718R0::A &, const DefaultA &)' lvalue Function {{.*}} 'foo2' 'A (const P2718R0::A &, const DefaultA &)' (FunctionTemplate {{.*}} 'foo2')
// CHECK-NEXT: |     |   |           |       |       |-MaterializeTemporaryExpr {{.*}} 'const A':'const P2718R0::A' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |           |       |       | `-ImplicitCastExpr {{.*}} 'const A':'const P2718R0::A' <NoOp>
// CHECK-NEXT: |     |   |           |       |       |   `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |           |       |       |     `-CXXTemporaryObjectExpr {{.*}} 'A':'P2718R0::A' 'void ()'
// CHECK-NEXT: |     |   |           |       |       `-CXXDefaultArgExpr {{.*}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK-NEXT: |     |   |           |       |         `-MaterializeTemporaryExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |           |       |           `-ImplicitCastExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK-NEXT: |     |   |           |       |             `-CXXBindTemporaryExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |           |       |               `-CXXTemporaryObjectExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK-NEXT: |     |   |           |       `-CXXDefaultArgExpr {{.*}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK-NEXT: |     |   |           |         `-MaterializeTemporaryExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |           |           `-ImplicitCastExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK-NEXT: |     |   |           |             `-CXXBindTemporaryExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |           |               `-CXXTemporaryObjectExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK-NEXT: |     |   |           `-CXXDefaultArgExpr {{.*}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK-NEXT: |     |   |             `-MaterializeTemporaryExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   |               `-ImplicitCastExpr {{.*}} 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK-NEXT: |     |   |                 `-CXXBindTemporaryExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |                   `-CXXTemporaryObjectExpr {{.*}} 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK-NEXT: |     |   `-typeDetails: LValueReferenceType {{.*}} 'int (&)[3]'
// CHECK-NEXT: |     |     `-typeDetails: ParenType {{.*}} 'int[3]' sugar
// CHECK-NEXT: |     |       `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __begin1 'int *' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |     |   | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: DecayedType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |       |-typeDetails: ParenType {{.*}} 'int[3]' sugar
// CHECK-NEXT: |     |       | `-typeDetails: ConstantArrayType {{.*}} 'int[3]' 3
// CHECK-NEXT: |     |       |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |       `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __end1 'int *' cinit
// CHECK-NEXT: |     |   |-BinaryOperator {{.*}} 'int *' '+'
// CHECK-NEXT: |     |   | |-ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: |     |   | | `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} '__range1' 'int (&)[3]'
// CHECK: |     |   `-typeDetails: AutoType {{.*}} 'int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: PointerType {{.*}} 'int *'
// CHECK-NEXT: |     |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT: |     | |-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     | | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__end1' 'int *'
// CHECK-NEXT: |     |-UnaryOperator {{.*}} 'int *' lvalue prefix '++'
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} used e 'int' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |     |   | `-UnaryOperator {{.*}} 'int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} '__begin1' 'int *'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'int' sugar
// CHECK-NEXT: |     |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     `-CallExpr {{.*}} 'void'
// CHECK-NEXT: |       |-ImplicitCastExpr {{.*}} 'void (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: |       | `-DeclRefExpr {{.*}} 'void (int)' lvalue Function {{.*}} 'bar' 'void (int)'
// CHECK-NEXT: |       `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |         `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'e' 'int'

// Test member call && dependent context
void test13() {

  for (auto e : dg<A>().r().g().r().g().r().g())
    bar(e);
}

// CHECK: |-FunctionDecl {{.*}} test13 'void ()'
// CHECK-NEXT: | `-CompoundStmt {{.*}} 
// CHECK-NEXT: |   `-CXXForRangeStmt {{.*}} 
// CHECK-NEXT: |     |-<<<NULL>>>
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __range1 'A &&' cinit
// CHECK-NEXT: |     |   |-ExprWithCleanups {{.*}} 'A':'P2718R0::A' xvalue
// CHECK-NEXT: |     |   | `-MaterializeTemporaryExpr {{.*}} 'A':'P2718R0::A' xvalue extended by Var {{.*}} '__range1' 'A &&'
// CHECK-NEXT: |     |   |   `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |     `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A'
// CHECK-NEXT: |     |   |       `-MemberExpr {{.*}} '<bound member function type>' .g {{.*}}
// CHECK-NEXT: |     |   |         `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A' lvalue
// CHECK-NEXT: |     |   |           `-MemberExpr {{.*}} '<bound member function type>' .r {{.*}}
// CHECK-NEXT: |     |   |             `-MaterializeTemporaryExpr {{.*}} 'A':'P2718R0::A' xvalue extended by Var {{.*}} '__range1' 'A &&'
// CHECK-NEXT: |     |   |               `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |                 `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A'
// CHECK-NEXT: |     |   |                   `-MemberExpr {{.*}} '<bound member function type>' .g {{.*}}
// CHECK-NEXT: |     |   |                     `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A' lvalue
// CHECK-NEXT: |     |   |                       `-MemberExpr {{.*}} '<bound member function type>' .r {{.*}}
// CHECK-NEXT: |     |   |                         `-MaterializeTemporaryExpr {{.*}} 'A':'P2718R0::A' xvalue extended by Var {{.*}} '__range1' 'A &&'
// CHECK-NEXT: |     |   |                           `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |                             `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A'
// CHECK-NEXT: |     |   |                               `-MemberExpr {{.*}} '<bound member function type>' .g {{.*}}
// CHECK-NEXT: |     |   |                                 `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A' lvalue
// CHECK-NEXT: |     |   |                                   `-MemberExpr {{.*}} '<bound member function type>' .r {{.*}}
// CHECK-NEXT: |     |   |                                     `-MaterializeTemporaryExpr {{.*}} 'P2718R0::A' xvalue extended by Var {{.*}} '__range1' 'A &&'
// CHECK-NEXT: |     |   |                                       `-CXXBindTemporaryExpr {{.*}} 'P2718R0::A' (CXXTemporary {{.*}})
// CHECK-NEXT: |     |   |                                         `-CallExpr {{.*}} 'P2718R0::A'
// CHECK-NEXT: |     |   |                                           `-ImplicitCastExpr {{.*}} 'P2718R0::A (*)()' <FunctionToPointerDecay>
// CHECK-NEXT: |     |   |                                             `-DeclRefExpr {{.*}} 'P2718R0::A ()' lvalue Function {{.*}} 'dg' 'P2718R0::A ()' (FunctionTemplate {{.*}} 'dg')
// CHECK-NEXT: |     |   `-typeDetails: RValueReferenceType {{.*}} 'A &&'
// CHECK-NEXT: |     |     `-typeDetails: AutoType {{.*}} 'A' sugar
// CHECK-NEXT: |     |       `-typeDetails: ElaboratedType {{.*}} 'A' sugar
// CHECK-NEXT: |     |         `-typeDetails: RecordType {{.*}} 'P2718R0::A'
// CHECK-NEXT: |     |           `-CXXRecord {{.*}} 'A'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __begin1 'const int *' cinit
// CHECK-NEXT: |     |   |-CXXMemberCallExpr {{.*}} 'const int *'
// CHECK-NEXT: |     |   | `-MemberExpr {{.*}} '<bound member function type>' .begin {{.*}}
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'const P2718R0::A' lvalue <NoOp>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'A':'P2718R0::A' lvalue Var {{.*}} '__range1' 'A &&'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'const int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT: |     |       `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __end1 'const int *' cinit
// CHECK-NEXT: |     |   |-CXXMemberCallExpr {{.*}} 'const int *'
// CHECK-NEXT: |     |   | `-MemberExpr {{.*}} '<bound member function type>' .end {{.*}}
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'const P2718R0::A' lvalue <NoOp>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'A':'P2718R0::A' lvalue Var {{.*}} '__range1' 'A &&'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'const int *' sugar
// CHECK-NEXT: |     |     `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT: |     |       `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT: |     |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT: |     | |-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT: |     | | `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__end1' 'const int *'
// CHECK-NEXT: |     |-UnaryOperator {{.*}} 'const int *' lvalue prefix '++'
// CHECK-NEXT: |     | `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT: |     |-DeclStmt {{.*}} 
// CHECK-NEXT: |     | `-VarDecl {{.*}} used e 'int' cinit
// CHECK-NEXT: |     |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |     |   | `-UnaryOperator {{.*}} 'const int' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     |   |   `-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT: |     |   |     `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT: |     |   `-typeDetails: AutoType {{.*}} 'int' sugar
// CHECK-NEXT: |     |     `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     `-CallExpr {{.*}} 'void'
// CHECK-NEXT: |       |-ImplicitCastExpr {{.*}} 'void (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: |       | `-DeclRefExpr {{.*}} 'void (int)' lvalue Function {{.*}} 'bar' 'void (int)'
// CHECK-NEXT: |       `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: |         `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'e' 'int'


extern "C" void exit(int);

// CHECK: |-LinkageSpecDecl {{.*}} C
// CHECK-NEXT: | `-FunctionDecl {{.*}} used exit 'void (int)'
// CHECK-NEXT: |   `-ParmVarDecl {{.*}} 'int'
// CHECK-NEXT: |     `-typeDetails: BuiltinType {{.*}} 'int'

struct A14 {
  int arr[1];
  ~A14() noexcept(false) { throw 42; }
};

// CHECK: |-CXXRecordDecl {{.*}} referenced struct A14 definition
// CHECK-NEXT: | |-DefinitionData aggregate standard_layout has_constexpr_non_copy_move_ctor
// CHECK-NEXT: | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// CHECK-NEXT: | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | |-MoveConstructor
// CHECK-NEXT: | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | |-MoveAssignment
// CHECK-NEXT: | | `-Destructor non_trivial user_declared
// CHECK-NEXT: | |-CXXRecordDecl {{.*}} implicit referenced struct A14
// CHECK-NEXT: | |-FieldDecl {{.*}} referenced arr 'int[1]'
// CHECK-NEXT: | `-CXXDestructorDecl {{.*}} used ~A14 'void () noexcept(false)' implicit-inline
// CHECK-NEXT: |   |-CompoundStmt {{.*}} 
// CHECK-NEXT: |   | `-CXXThrowExpr {{.*}} 'void'
// CHECK-NEXT: |   |   `-IntegerLiteral {{.*}} 'int' 42
// CHECK-NEXT: |   `-attrDetails: InferredNoReturnAttr {{.*}} <<invalid sloc>> Implicit

struct B14 {
  int x;
  const A14 &a = A14{{0}};
  const int *begin() { return a.arr; }
  const int *end() { return &a.arr[1]; }
};

void test14() {
  // The ExprWithCleanups in CXXDefaultInitExpr will be ignored.
  for (auto &&x : B14{0}.a.arr) { exit(0); }
  for (auto &&x : B14{0}) { exit(0); }
}
} // namespace P2718R0

// CHECK: |-CXXRecordDecl {{.*}} referenced struct B14 definition
// CHECK-NEXT: | |-DefinitionData pass_in_registers aggregate trivially_copyable literal has_constexpr_non_copy_move_ctor
// CHECK-NEXT: | | |-DefaultConstructor exists non_trivial constexpr needs_implicit defaulted_is_constexpr
// CHECK-NEXT: | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT: | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT: | | |-MoveAssignment exists trivial needs_implicit
// CHECK-NEXT: | | `-Destructor simple irrelevant trivial constexpr
// CHECK-NEXT: | |-CXXRecordDecl {{.*}} implicit struct B14
// CHECK-NEXT: | |-FieldDecl {{.*}} x 'int'
// CHECK-NEXT: | |-FieldDecl {{.*}} referenced a 'const A14 &'
// CHECK-NEXT: | | `-ExprWithCleanups {{.*}} 'const A14':'const P2718R0::A14' lvalue
// CHECK-NEXT: | |   `-MaterializeTemporaryExpr {{.*}} 'const A14':'const P2718R0::A14' lvalue extended by Field {{.*}} 'a' 'const A14 &'
// CHECK-NEXT: | |     `-ImplicitCastExpr {{.*}} 'const A14':'const P2718R0::A14' <NoOp>
// CHECK-NEXT: | |       `-CXXFunctionalCastExpr {{.*}} 'A14':'P2718R0::A14' functional cast to A14 <NoOp>
// CHECK-NEXT: | |         `-CXXBindTemporaryExpr {{.*}} 'A14':'P2718R0::A14' (CXXTemporary {{.*}})
// CHECK-NEXT: | |           `-InitListExpr {{.*}} 'A14':'P2718R0::A14'
// CHECK-NEXT: | |             `-InitListExpr {{.*}} 'int[1]'
// CHECK-NEXT: | |               `-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: | |-CXXMethodDecl {{.*}} used begin 'const int *()' implicit-inline
// CHECK-NEXT: | | `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |   `-ReturnStmt {{.*}} 
// CHECK-NEXT: | |     `-ImplicitCastExpr {{.*}} 'const int *' <ArrayToPointerDecay>
// CHECK-NEXT: | |       `-MemberExpr {{.*}} 'const int[1]' lvalue .arr {{.*}}
// CHECK-NEXT: | |         `-MemberExpr {{.*}} 'const A14':'const P2718R0::A14' lvalue ->a {{.*}}
// CHECK-NEXT: | |           `-CXXThisExpr {{.*}} 'P2718R0::B14 *' implicit this
// CHECK-NEXT: | |-CXXMethodDecl {{.*}} used end 'const int *()' implicit-inline
// CHECK-NEXT: | | `-CompoundStmt {{.*}} 
// CHECK-NEXT: | |   `-ReturnStmt {{.*}} 
// CHECK-NEXT: | |     `-UnaryOperator {{.*}} 'const int *' prefix '&' cannot overflow
// CHECK-NEXT: | |       `-ArraySubscriptExpr {{.*}} 'const int' lvalue
// CHECK-NEXT: | |         |-ImplicitCastExpr {{.*}} 'const int *' <ArrayToPointerDecay>
// CHECK-NEXT: | |         | `-MemberExpr {{.*}} 'const int[1]' lvalue .arr {{.*}}
// CHECK-NEXT: | |         |   `-MemberExpr {{.*}} 'const A14':'const P2718R0::A14' lvalue ->a {{.*}}
// CHECK-NEXT: | |         |     `-CXXThisExpr {{.*}} 'P2718R0::B14 *' implicit this
// CHECK-NEXT: | |         `-IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: | `-CXXDestructorDecl {{.*}} implicit referenced constexpr ~B14 'void () noexcept' inline default trivial
// CHECK-NEXT: `-FunctionDecl {{.*}} test14 'void ()'
// CHECK-NEXT:   `-CompoundStmt {{.*}} 
// CHECK-NEXT:     |-CXXForRangeStmt {{.*}} 
// CHECK-NEXT:     | |-<<<NULL>>>
// CHECK-NEXT:     | |-DeclStmt {{.*}} 
// CHECK-NEXT:     | | `-VarDecl {{.*}} implicit used __range1 'const int (&)[1]' cinit
// CHECK-NEXT:     | |   |-ExprWithCleanups {{.*}} 'const int[1]' lvalue
// CHECK-NEXT:     | |   | `-MemberExpr {{.*}} 'const int[1]' lvalue .arr {{.*}}
// CHECK-NEXT:     | |   |   `-MemberExpr {{.*}} 'const A14':'const P2718R0::A14' lvalue .a {{.*}}
// CHECK-NEXT:     | |   |     `-MaterializeTemporaryExpr {{.*}} 'B14':'P2718R0::B14' xvalue extended by Var {{.*}} '__range1' 'const int (&)[1]'
// CHECK-NEXT:     | |   |       `-CXXFunctionalCastExpr {{.*}} 'B14':'P2718R0::B14' functional cast to B14 <NoOp>
// CHECK-NEXT:     | |   |         `-InitListExpr {{.*}} 'B14':'P2718R0::B14'
// CHECK-NEXT:     | |   |           |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT:     | |   |           `-CXXDefaultInitExpr {{.*}} 'const A14':'const P2718R0::A14' lvalue has rewritten init
// CHECK-NEXT:     | |   |             `-MaterializeTemporaryExpr {{.*}} 'const A14':'const P2718R0::A14' lvalue extended by Var {{.*}} '__range1' 'const int (&)[1]'
// CHECK-NEXT:     | |   |               `-ImplicitCastExpr {{.*}} 'const A14':'const P2718R0::A14' <NoOp>
// CHECK-NEXT:     | |   |                 `-CXXFunctionalCastExpr {{.*}} 'A14':'P2718R0::A14' functional cast to A14 <NoOp>
// CHECK-NEXT:     | |   |                   `-CXXBindTemporaryExpr {{.*}} 'A14':'P2718R0::A14' (CXXTemporary {{.*}})
// CHECK-NEXT:     | |   |                     `-InitListExpr {{.*}} 'A14':'P2718R0::A14'
// CHECK-NEXT:     | |   |                       `-InitListExpr {{.*}} 'int[1]'
// CHECK-NEXT:     | |   |                         `-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT:     | |   `-typeDetails: LValueReferenceType {{.*}} 'const int (&)[1]'
// CHECK-NEXT:     | |     `-qualTypeDetail: QualType {{.*}} 'const int[1]' const
// CHECK-NEXT:     | |       `-typeDetails: ConstantArrayType {{.*}} 'int[1]' 1
// CHECK-NEXT:     | |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:     | |-DeclStmt {{.*}} 
// CHECK-NEXT:     | | `-VarDecl {{.*}} implicit used __begin1 'const int *' cinit
// CHECK-NEXT:     | |   |-ImplicitCastExpr {{.*}} 'const int *' <ArrayToPointerDecay>
// CHECK-NEXT:     | |   | `-DeclRefExpr {{.*}} 'const int[1]' lvalue Var {{.*}} '__range1' 'const int (&)[1]'
// CHECK-NEXT:     | |   `-typeDetails: AutoType {{.*}} 'const int *' sugar
// CHECK-NEXT:     | |     `-typeDetails: DecayedType {{.*}} 'const int *' sugar
// CHECK-NEXT:     | |       |-qualTypeDetail: QualType {{.*}} 'const int[1]' const
// CHECK-NEXT:     | |       | `-typeDetails: ConstantArrayType {{.*}} 'int[1]' 1
// CHECK-NEXT:     | |       |   `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:     | |       `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT:     | |         `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT:     | |           `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:     | |-DeclStmt {{.*}} 
// CHECK-NEXT:     | | `-VarDecl {{.*}} implicit used __end1 'const int *' cinit
// CHECK-NEXT:     | |   |-BinaryOperator {{.*}} 'const int *' '+'
// CHECK-NEXT:     | |   | |-ImplicitCastExpr {{.*}} 'const int *' <ArrayToPointerDecay>
// CHECK:     | |   | | `-DeclRefExpr {{.*}} 'const int[1]' lvalue Var {{.*}} '__range1' 'const int (&)[1]'
// CHECK:     | |   `-typeDetails: AutoType {{.*}} 'const int *' sugar
// CHECK-NEXT:     | |     `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT:     | |       `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT:     | |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:     | |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT:     | | |-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT:     | | | `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT:     | | `-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT:     | |   `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__end1' 'const int *'
// CHECK-NEXT:     | |-UnaryOperator {{.*}} 'const int *' lvalue prefix '++'
// CHECK-NEXT:     | | `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT:     | |-DeclStmt {{.*}} 
// CHECK-NEXT:     | | `-VarDecl {{.*}} x 'const int &' cinit
// CHECK-NEXT:     | |   |-UnaryOperator {{.*}} 'const int' lvalue prefix '*' cannot overflow
// CHECK-NEXT:     | |   | `-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT:     | |   |   `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT:     | |   `-typeDetails: LValueReferenceType {{.*}} 'const int &'
// CHECK-NEXT:     | |     `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT:     | |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:     | `-CompoundStmt {{.*}} 
// CHECK-NEXT:     |   `-CallExpr {{.*}} 'void'
// CHECK-NEXT:     |     |-ImplicitCastExpr {{.*}} 'void (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT:     |     | `-DeclRefExpr {{.*}} 'void (int)' lvalue Function {{.*}} 'exit' 'void (int)'
// CHECK-NEXT:     |     `-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT:     `-CXXForRangeStmt {{.*}} 
// CHECK-NEXT:       |-<<<NULL>>>
// CHECK-NEXT:       |-DeclStmt {{.*}} 
// CHECK-NEXT:       | `-VarDecl {{.*}} implicit used __range1 'B14 &&' cinit
// CHECK-NEXT:       |   |-ExprWithCleanups {{.*}} 'B14':'P2718R0::B14' xvalue
// CHECK-NEXT:       |   | `-MaterializeTemporaryExpr {{.*}} 'B14':'P2718R0::B14' xvalue extended by Var {{.*}} '__range1' 'B14 &&'
// CHECK-NEXT:       |   |   `-CXXFunctionalCastExpr {{.*}} 'B14':'P2718R0::B14' functional cast to B14 <NoOp>
// CHECK-NEXT:       |   |     `-InitListExpr {{.*}} 'B14':'P2718R0::B14'
// CHECK-NEXT:       |   |       |-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT:       |   |       `-CXXDefaultInitExpr {{.*}} 'const A14':'const P2718R0::A14' lvalue has rewritten init
// CHECK-NEXT:       |   |         `-MaterializeTemporaryExpr {{.*}} 'const A14':'const P2718R0::A14' lvalue extended by Var {{.*}} '__range1' 'B14 &&'
// CHECK-NEXT:       |   |           `-ImplicitCastExpr {{.*}} 'const A14':'const P2718R0::A14' <NoOp>
// CHECK-NEXT:       |   |             `-CXXFunctionalCastExpr {{.*}} 'A14':'P2718R0::A14' functional cast to A14 <NoOp>
// CHECK-NEXT:       |   |               `-CXXBindTemporaryExpr {{.*}} 'A14':'P2718R0::A14' (CXXTemporary {{.*}})
// CHECK-NEXT:       |   |                 `-InitListExpr {{.*}} 'A14':'P2718R0::A14'
// CHECK-NEXT:       |   |                   `-InitListExpr {{.*}} 'int[1]'
// CHECK-NEXT:       |   |                     `-IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT:       |   `-typeDetails: RValueReferenceType {{.*}} 'B14 &&'
// CHECK-NEXT:       |     `-typeDetails: AutoType {{.*}} 'B14' sugar
// CHECK-NEXT:       |       `-typeDetails: ElaboratedType {{.*}} 'B14' sugar
// CHECK-NEXT:       |         `-typeDetails: RecordType {{.*}} 'P2718R0::B14'
// CHECK-NEXT:       |           `-CXXRecord {{.*}} 'B14'
// CHECK-NEXT:       |-DeclStmt {{.*}} 
// CHECK-NEXT:       | `-VarDecl {{.*}} implicit used __begin1 'const int *' cinit
// CHECK-NEXT:       |   |-CXXMemberCallExpr {{.*}} 'const int *'
// CHECK-NEXT:       |   | `-MemberExpr {{.*}} '<bound member function type>' .begin {{.*}}
// CHECK-NEXT:       |   |   `-DeclRefExpr {{.*}} 'B14':'P2718R0::B14' lvalue Var {{.*}} '__range1' 'B14 &&'
// CHECK-NEXT:       |   `-typeDetails: AutoType {{.*}} 'const int *' sugar
// CHECK-NEXT:       |     `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT:       |       `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT:       |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:       |-DeclStmt {{.*}} 
// CHECK-NEXT:       | `-VarDecl {{.*}} implicit used __end1 'const int *' cinit
// CHECK-NEXT:       |   |-CXXMemberCallExpr {{.*}} 'const int *'
// CHECK-NEXT:       |   | `-MemberExpr {{.*}} '<bound member function type>' .end {{.*}}
// CHECK-NEXT:       |   |   `-DeclRefExpr {{.*}} 'B14':'P2718R0::B14' lvalue Var {{.*}} '__range1' 'B14 &&'
// CHECK-NEXT:       |   `-typeDetails: AutoType {{.*}} 'const int *' sugar
// CHECK-NEXT:       |     `-typeDetails: PointerType {{.*}} 'const int *'
// CHECK-NEXT:       |       `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT:       |         `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:       |-BinaryOperator {{.*}} 'bool' '!='
// CHECK-NEXT:       | |-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT:       | | `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT:       | `-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT:       |   `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__end1' 'const int *'
// CHECK-NEXT:       |-UnaryOperator {{.*}} 'const int *' lvalue prefix '++'
// CHECK-NEXT:       | `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT:       |-DeclStmt {{.*}} 
// CHECK-NEXT:       | `-VarDecl {{.*}} x 'const int &' cinit
// CHECK-NEXT:       |   |-UnaryOperator {{.*}} 'const int' lvalue prefix '*' cannot overflow
// CHECK-NEXT:       |   | `-ImplicitCastExpr {{.*}} 'const int *' <LValueToRValue>
// CHECK-NEXT:       |   |   `-DeclRefExpr {{.*}} 'const int *' lvalue Var {{.*}} '__begin1' 'const int *'
// CHECK-NEXT:       |   `-typeDetails: LValueReferenceType {{.*}} 'const int &'
// CHECK-NEXT:       |     `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK-NEXT:       |       `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK-NEXT:       `-CompoundStmt {{.*}} 
// CHECK-NEXT:         `-CallExpr {{.*}} 'void'
// CHECK-NEXT:           |-ImplicitCastExpr {{.*}} 'void (*)(int)' <FunctionToPointerDecay>
// CHECK-NEXT:           | `-DeclRefExpr {{.*}} 'void (int)' lvalue Function {{.*}} 'exit' 'void (int)'
// CHECK-NEXT:           `-IntegerLiteral {{.*}} 'int' 0
