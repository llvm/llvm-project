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













































































struct B : A {};
int (&f(const A *))[3];
const A *g(const A &);
void bar(int) {}

void test2() {
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  for (auto e : f(g(B())))
    bar(e);
}

// Test discard statement.
struct LockGuard {
    LockGuard() {}
    ~LockGuard() {}
};

void test3() {
  int v[] = {42, 17, 13};

  
  
  
  
  
  
  
  
  
  
  
  
  for ([[maybe_unused]] int x : static_cast<void>(LockGuard()), v)
    LockGuard guard;

  
  
  
  
  
  
  
  
  
  
  
  for ([[maybe_unused]] int x : (void)LockGuard(), v)
    LockGuard guard;

  
  
  
  
  
  
  
  
  
  
  for ([[maybe_unused]] int x : LockGuard(), v)
    LockGuard guard;
}

// Test default arg
int (&default_arg_fn(const A & = A()))[3];
void test4() {

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  for (auto e : default_arg_fn()) 
    bar(e);
}

struct DefaultA {
  DefaultA() {}
  ~DefaultA() {}
};

A foo(const A&, const DefaultA &Default = DefaultA()) {
  return A();
}

void test5() {
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  for (auto e : default_arg_fn(foo(foo(foo(A())))))
    bar(e);
}

struct C : public A {
  C() {}
  C(int, const C &, const DefaultA & = DefaultA()) {}
};

void test6() {
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  for (auto e : C(0, C(0, C(0, C()))))
    bar(e);
}

// Test member call
void test7() {
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  for (auto e : g().r().g().r().g().r().g())
    bar(e);
}

// Test basic && dependent context
template <typename T> T dg() { return T(); }
template <typename T> const T &df1(const T &t) { return t; }

void test8() {
  [[maybe_unused]] int sum = 0;
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  for (auto e : df1(dg<A>()))
    sum += e;
}

template <typename T> int (&df2(const T *))[3];
const A *dg2(const A &);

void test9() {
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  for (auto e : df2(dg2(B())))
    bar(e);
}

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

// Test default argument && dependent context
template <typename T> int (&default_arg_fn2(const T & = T()))[3];
void test11() {
  for (auto e : default_arg_fn2<A>()) 
    bar(e);
}

template <typename T> A foo2(const T&, const DefaultA &Default = DefaultA());

void test12() {
  for (auto e : default_arg_fn2(foo2(foo2(foo2(A())))))
    bar(e);
}

// Test member call && dependent context
void test13() {

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  for (auto e : dg<A>().r().g().r().g().r().g())
    bar(e);
}
} // namespace P2718R0

// CHECK: TranslationUnitDecl 0x{{.+}} <<invalid sloc>> <invalid sloc>
// CHECK: |-TypedefDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> implicit __int128_t '__int128'
// CHECK: | `-typeDetails: BuiltinType 0x{{.+}} '__int128'
// CHECK: |-TypedefDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> implicit __uint128_t 'unsigned __int128'
// CHECK: | `-typeDetails: BuiltinType 0x{{.+}} 'unsigned __int128'
// CHECK: |-TypedefDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> implicit __NSConstantString '__NSConstantString_tag'
// CHECK: | `-typeDetails: RecordType 0x{{.+}} '__NSConstantString_tag'
// CHECK: |   `-CXXRecord 0x{{.+}} '__NSConstantString_tag'
// CHECK: |-TypedefDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> implicit __builtin_ms_va_list 'char *'
// CHECK: | `-typeDetails: PointerType 0x{{.+}} 'char *'
// CHECK: |   `-typeDetails: BuiltinType 0x{{.+}} 'char'
// CHECK: |-TypedefDecl 0x{{.+}} <<invalid sloc>> <invalid sloc> implicit __builtin_va_list '__va_list_tag[1]'
// CHECK: | `-typeDetails: ConstantArrayType 0x{{.+}} '__va_list_tag[1]' 1
// CHECK: |   `-typeDetails: RecordType 0x{{.+}} '__va_list_tag'
// CHECK: |     `-CXXRecord 0x{{.+}} '__va_list_tag'
// CHECK: `-NamespaceDecl 0x{{.+}} <{{.*}} line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} P2718R0
// CHECK:   |-CXXRecordDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} referenced struct A definition
// CHECK:   | |-DefinitionData standard_layout has_user_declared_ctor can_const_default_init
// CHECK:   | | |-DefaultConstructor exists non_trivial user_provided defaulted_is_constexpr
// CHECK:   | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK:   | | |-MoveConstructor
// CHECK:   | | |-CopyAssignment simple trivial has_const_param implicit_has_const_param
// CHECK:   | | |-MoveAssignment
// CHECK:   | | `-Destructor non_trivial user_declared
// CHECK:   | |-CXXRecordDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit referenced struct A
// CHECK:   | |-FieldDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} referenced a 'int[3]'
// CHECK:   | | `-InitListExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]'
// CHECK:   | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK:   | |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 2
// CHECK:   | |   `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 3
// CHECK:   | |-CXXConstructorDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used A 'void ()' implicit-inline
// CHECK:   | | |-CXXCtorInitializer Field 0x{{.+}} 'a' 'int[3]'
// CHECK:   | | | `-CXXDefaultInitExpr 0x{{.+}} <col:{{.*}}> 'int[3]' has rewritten init
// CHECK:   | | |   `-InitListExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'int[3]'
// CHECK:   | | |     |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 1
// CHECK:   | | |     |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 2
// CHECK:   | | |     `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 3
// CHECK:   | | `-CompoundStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:   | |-CXXDestructorDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used ~A 'void () noexcept' implicit-inline
// CHECK:   | | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |-CXXMethodDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used begin 'const int *() const' implicit-inline
// CHECK:   | | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |   `-ReturnStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |     `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <ArrayToPointerDecay>
// CHECK:   | |       `-MemberExpr 0x{{.+}} <col:{{.*}}> 'const int[3]' lvalue ->a 0x{{.+}}
// CHECK:   | |         `-CXXThisExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::A *' implicit this
// CHECK:   | |-CXXMethodDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used end 'const int *() const' implicit-inline
// CHECK:   | | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |   `-ReturnStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |     `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const int *' '+'
// CHECK:   | |       |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <ArrayToPointerDecay>
// CHECK:   | |       | `-MemberExpr 0x{{.+}} <col:{{.*}}> 'const int[3]' lvalue ->a 0x{{.+}}
// CHECK:   | |       |   `-CXXThisExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::A *' implicit this
// CHECK:   | |       `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 3
// CHECK:   | |-CXXMethodDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used r 'A &()' implicit-inline
// CHECK:   | | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |   `-ReturnStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |     `-UnaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A' lvalue prefix '*' cannot overflow
// CHECK:   | |       `-CXXThisExpr 0x{{.+}} <col:{{.*}}> 'P2718R0::A *' this
// CHECK:   | |-CXXMethodDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used g 'A ()' implicit-inline
// CHECK:   | | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |   `-ReturnStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |     `-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:   | |       `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   | |         `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' 'void ()'
// CHECK:   | |-CXXConstructorDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit constexpr A 'void (const A &)' inline default trivial noexcept-unevaluated 0x{{.+}}
// CHECK:   | | `-ParmVarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 'const A &'
// CHECK:   | |   `-typeDetails: LValueReferenceType 0x{{.+}} 'const A &'
// CHECK:   | |     `-qualTypeDetail: QualType 0x{{.+}} 'const A' const
// CHECK:   | |       `-typeDetails: ElaboratedType 0x{{.+}} 'A' sugar
// CHECK:   | |         `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   | |           `-CXXRecord 0x{{.+}} 'A'
// CHECK:   | `-CXXMethodDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit constexpr operator= 'A &(const A &)' inline default trivial noexcept-unevaluated 0x{{.+}}
// CHECK:   |   `-ParmVarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 'const A &'
// CHECK:   |     `-typeDetails: LValueReferenceType 0x{{.+}} 'const A &'
// CHECK:   |       `-qualTypeDetail: QualType 0x{{.+}} 'const A' const
// CHECK:   |         `-typeDetails: ElaboratedType 0x{{.+}} 'A' sugar
// CHECK:   |           `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   |             `-CXXRecord 0x{{.+}} 'A'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used g 'A ()'
// CHECK:   | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |   `-ReturnStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |     `-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:   |       `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |         `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' 'void ()'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used f1 'const A &(const A &)'
// CHECK:   | |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used t 'const A &'
// CHECK:   | | `-typeDetails: LValueReferenceType 0x{{.+}} 'const A &'
// CHECK:   | |   `-qualTypeDetail: QualType 0x{{.+}} 'const A' const
// CHECK:   | |     `-typeDetails: ElaboratedType 0x{{.+}} 'A' sugar
// CHECK:   | |       `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   | |         `-CXXRecord 0x{{.+}} 'A'
// CHECK:   | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |   `-ReturnStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const A':'const P2718R0::A' lvalue ParmVar 0x{{.+}} 't' 'const A &'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test1 'void ()'
// CHECK:   | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:   |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used sum 'int' cinit
// CHECK:   |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:   |   |   |-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   |   `-attrDetails: UnusedAttr 0x{{.+}} <col:{{.*}}> maybe_unused
// CHECK:   |   `-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |     |-<<<NULL>>>
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'const A &' cinit
// CHECK:   |     |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' lvalue
// CHECK:   |     |   | `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' lvalue
// CHECK:   |     |   |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const A &(*)(const A &)' <FunctionToPointerDecay>
// CHECK:   |     |   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const A &(const A &)' lvalue Function 0x{{.+}} 'f1' 'const A &(const A &)'
// CHECK:   |     |   |   `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' lvalue extended by Var 0x{{.+}} '__range1' 'const A &'
// CHECK:   |     |   |     `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' <NoOp>
// CHECK:   |     |   |       `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |         `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:   |     |   |           `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'A (*)()' <FunctionToPointerDecay>
// CHECK:   |     |   |             `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'A ()' lvalue Function 0x{{.+}} 'g' 'A ()'
// CHECK:   |     |   `-typeDetails: LValueReferenceType 0x{{.+}} 'const A &'
// CHECK:   |     |     `-qualTypeDetail: QualType 0x{{.+}} 'const A' const
// CHECK:   |     |       `-typeDetails: ElaboratedType 0x{{.+}} 'A' sugar
// CHECK:   |     |         `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   |     |           `-CXXRecord 0x{{.+}} 'A'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'const int *' cinit
// CHECK:   |     |   |-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}> 'const int *'
// CHECK:   |     |   | `-MemberExpr 0x{{.+}} <col:{{.*}}> '<bound member function type>' .begin 0x{{.+}}
// CHECK:   |     |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const A':'const P2718R0::A' lvalue Var 0x{{.+}} '__range1' 'const A &'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'const int *' sugar
// CHECK:   |     |     `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:   |     |       `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __end1 'const int *' cinit
// CHECK:   |     |   |-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}> 'const int *'
// CHECK:   |     |   | `-MemberExpr 0x{{.+}} <col:{{.*}}> '<bound member function type>' .end 0x{{.+}}
// CHECK:   |     |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const A':'const P2718R0::A' lvalue Var 0x{{.+}} '__range1' 'const A &'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'const int *' sugar
// CHECK:   |     |     `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:   |     |       `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:   |     | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <LValueToRValue>
// CHECK:   |     | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__begin1' 'const int *'
// CHECK:   |     | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <LValueToRValue>
// CHECK:   |     |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__end1' 'const int *'
// CHECK:   |     |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'const int *' lvalue prefix '++'
// CHECK:   |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__begin1' 'const int *'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used e 'int' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |     |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'const int' lvalue prefix '*' cannot overflow
// CHECK:   |     |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <LValueToRValue>
// CHECK:   |     |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__begin1' 'const int *'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int' sugar
// CHECK:   |     |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     `-CompoundAssignOperator 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'int' lvalue '+=' ComputeLHSTy='int' ComputeResultTy='int'
// CHECK:   |       |-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'sum' 'int'
// CHECK:   |       `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |         `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'e' 'int'
// CHECK:   |-CXXRecordDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} referenced struct B definition
// CHECK:   | |-DefinitionData aggregate standard_layout has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK:   | | |-DefaultConstructor exists non_trivial constexpr defaulted_is_constexpr
// CHECK:   | | |-CopyConstructor simple trivial has_const_param needs_overload_resolution implicit_has_const_param
// CHECK:   | | |-MoveConstructor exists simple trivial needs_overload_resolution
// CHECK:   | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:   | | |-MoveAssignment exists simple trivial needs_overload_resolution
// CHECK:   | | `-Destructor simple non_trivial constexpr needs_overload_resolution
// CHECK:   | |-public 'A':'P2718R0::A'
// CHECK:   | |-CXXRecordDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit struct B
// CHECK:   | |-CXXConstructorDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit constexpr B 'void (const B &)' inline default trivial noexcept-unevaluated 0x{{.+}}
// CHECK:   | | `-ParmVarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 'const B &'
// CHECK:   | |   `-typeDetails: LValueReferenceType 0x{{.+}} 'const B &'
// CHECK:   | |     `-qualTypeDetail: QualType 0x{{.+}} 'const B' const
// CHECK:   | |       `-typeDetails: ElaboratedType 0x{{.+}} 'B' sugar
// CHECK:   | |         `-typeDetails: RecordType 0x{{.+}} 'P2718R0::B'
// CHECK:   | |           `-CXXRecord 0x{{.+}} 'B'
// CHECK:   | |-CXXConstructorDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit constexpr B 'void (B &&)' inline default trivial noexcept-unevaluated 0x{{.+}}
// CHECK:   | | `-ParmVarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 'B &&'
// CHECK:   | |   `-typeDetails: RValueReferenceType 0x{{.+}} 'B &&'
// CHECK:   | |     `-typeDetails: ElaboratedType 0x{{.+}} 'B' sugar
// CHECK:   | |       `-typeDetails: RecordType 0x{{.+}} 'P2718R0::B'
// CHECK:   | |         `-CXXRecord 0x{{.+}} 'B'
// CHECK:   | |-CXXMethodDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit constexpr operator= 'B &(B &&)' inline default trivial noexcept-unevaluated 0x{{.+}}
// CHECK:   | | `-ParmVarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 'B &&'
// CHECK:   | |   `-typeDetails: RValueReferenceType 0x{{.+}} 'B &&'
// CHECK:   | |     `-typeDetails: ElaboratedType 0x{{.+}} 'B' sugar
// CHECK:   | |       `-typeDetails: RecordType 0x{{.+}} 'P2718R0::B'
// CHECK:   | |         `-CXXRecord 0x{{.+}} 'B'
// CHECK:   | |-CXXDestructorDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used constexpr ~B 'void () noexcept' inline default
// CHECK:   | | `-CompoundStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   | `-CXXConstructorDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used constexpr B 'void () noexcept(false)' inline default
// CHECK:   |   |-CXXCtorInitializer 'A':'P2718R0::A'
// CHECK:   |   | `-CXXConstructExpr 0x{{.+}} <col:{{.*}}> 'A':'P2718R0::A' 'void ()'
// CHECK:   |   `-CompoundStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used f 'int (&(const A *))[3]'
// CHECK:   | `-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} 'const A *'
// CHECK:   |   `-typeDetails: PointerType 0x{{.+}} 'const A *'
// CHECK:   |     `-qualTypeDetail: QualType 0x{{.+}} 'const A' const
// CHECK:   |       `-typeDetails: ElaboratedType 0x{{.+}} 'A' sugar
// CHECK:   |         `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   |           `-CXXRecord 0x{{.+}} 'A'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used g 'const A *(const A &)'
// CHECK:   | `-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} 'const A &'
// CHECK:   |   `-typeDetails: LValueReferenceType 0x{{.+}} 'const A &'
// CHECK:   |     `-qualTypeDetail: QualType 0x{{.+}} 'const A' const
// CHECK:   |       `-typeDetails: ElaboratedType 0x{{.+}} 'A' sugar
// CHECK:   |         `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   |           `-CXXRecord 0x{{.+}} 'A'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used bar 'void (int)'
// CHECK:   | |-ParmVarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 'int'
// CHECK:   | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test2 'void ()'
// CHECK:   | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   `-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |     |-<<<NULL>>>
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK:   |     |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |     |   | `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |     |   |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int (&(*)(const A *))[3]' <FunctionToPointerDecay>
// CHECK:   |     |   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int (&(const A *))[3]' lvalue Function 0x{{.+}} 'f' 'int (&(const A *))[3]'
// CHECK:   |     |   |   `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A *'
// CHECK:   |     |   |     |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const A *(*)(const A &)' <FunctionToPointerDecay>
// CHECK:   |     |   |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const A *(const A &)' lvalue Function 0x{{.+}} 'g' 'const A *(const A &)'
// CHECK:   |     |   |     `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' lvalue <DerivedToBase (A)>
// CHECK:   |     |   |       `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const B':'const P2718R0::B' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |         `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const B':'const P2718R0::B' <NoOp>
// CHECK:   |     |   |           `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'B':'P2718R0::B' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |             `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'B':'P2718R0::B' 'void () noexcept(false)' zeroing
// CHECK:   |     |   `-typeDetails: LValueReferenceType 0x{{.+}} 'int (&)[3]'
// CHECK:   |     |     `-typeDetails: ParenType 0x{{.+}} 'int[3]' sugar
// CHECK:   |     |       `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'int *' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |     |     `-typeDetails: DecayedType 0x{{.+}} 'int *' sugar
// CHECK:   |     |       |-typeDetails: ParenType 0x{{.+}} 'int[3]' sugar
// CHECK:   |     |       | `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |     |       |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |       `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __end1 'int *' cinit
// CHECK:   |     |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int *' '+'
// CHECK:   |     |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'long' 3
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |     |     `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |     |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:   |     | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__end1' 'int *'
// CHECK:   |     |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int *' lvalue prefix '++'
// CHECK:   |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used e 'int' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |     |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK:   |     |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int' sugar
// CHECK:   |     |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     `-CallExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'void'
// CHECK:   |       |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'void (*)(int)' <FunctionToPointerDecay>
// CHECK:   |       | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'void (int)' lvalue Function 0x{{.+}} 'bar' 'void (int)'
// CHECK:   |       `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |         `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'e' 'int'
// CHECK:   |-CXXRecordDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} referenced struct LockGuard definition
// CHECK:   | |-DefinitionData empty standard_layout has_user_declared_ctor can_const_default_init
// CHECK:   | | |-DefaultConstructor exists non_trivial user_provided defaulted_is_constexpr
// CHECK:   | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK:   | | |-MoveConstructor
// CHECK:   | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:   | | |-MoveAssignment
// CHECK:   | | `-Destructor non_trivial user_declared
// CHECK:   | |-CXXRecordDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit referenced struct LockGuard
// CHECK:   | |-CXXConstructorDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used LockGuard 'void ()' implicit-inline
// CHECK:   | | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |-CXXDestructorDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used ~LockGuard 'void () noexcept' implicit-inline
// CHECK:   | | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | `-CXXConstructorDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit constexpr LockGuard 'void (const LockGuard &)' inline default trivial noexcept-unevaluated 0x{{.+}}
// CHECK:   |   `-ParmVarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 'const LockGuard &'
// CHECK:   |     `-typeDetails: LValueReferenceType 0x{{.+}} 'const LockGuard &'
// CHECK:   |       `-qualTypeDetail: QualType 0x{{.+}} 'const LockGuard' const
// CHECK:   |         `-typeDetails: ElaboratedType 0x{{.+}} 'LockGuard' sugar
// CHECK:   |           `-typeDetails: RecordType 0x{{.+}} 'P2718R0::LockGuard'
// CHECK:   |             `-CXXRecord 0x{{.+}} 'LockGuard'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test3 'void ()'
// CHECK:   | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:   |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used v 'int[3]' cinit
// CHECK:   |   |   |-InitListExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]'
// CHECK:   |   |   | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 42
// CHECK:   |   |   | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 17
// CHECK:   |   |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 13
// CHECK:   |   |   `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |   |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   |-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   | |-<<<NULL>>>
// CHECK:   |   | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |   | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK:   |   | |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |   | |   | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue ','
// CHECK:   |   | |   |   |-CXXStaticCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'void' static_cast<void> <ToVoid>
// CHECK:   |   | |   |   | `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' xvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |   | |   |   |   `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' (CXXTemporary 0x{{.+}})
// CHECK:   |   | |   |   |     `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK:   |   | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} 'v' 'int[3]'
// CHECK:   |   | |   `-typeDetails: LValueReferenceType 0x{{.+}} 'int (&)[3]'
// CHECK:   |   | |     `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |   | |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |   | | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'int *' cinit
// CHECK:   |   | |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |   | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |   | |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |   | |     `-typeDetails: DecayedType 0x{{.+}} 'int *' sugar
// CHECK:   |   | |       |-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |   | |       | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |       `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |   | |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |   | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __end1 'int *' cinit
// CHECK:   |   | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int *' '+'
// CHECK:   |   | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |   | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |   | |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'long' 3
// CHECK:   |   | |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |   | |     `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |   | |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:   |   | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |   | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |   | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__end1' 'int *'
// CHECK:   |   | |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int *' lvalue prefix '++'
// CHECK:   |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |   | |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |   | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} x 'int' cinit
// CHECK:   |   | |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |   | |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK:   |   | |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |   | |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |   | |   |-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |   `-attrDetails: UnusedAttr 0x{{.+}} <col:{{.*}}> maybe_unused
// CHECK:   |   | `-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:   |   |   `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} guard 'LockGuard':'P2718R0::LockGuard' callinit destroyed
// CHECK:   |   |     |-CXXConstructExpr 0x{{.+}} <col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK:   |   |     `-typeDetails: ElaboratedType 0x{{.+}} 'LockGuard' sugar
// CHECK:   |   |       `-typeDetails: RecordType 0x{{.+}} 'P2718R0::LockGuard'
// CHECK:   |   |         `-CXXRecord 0x{{.+}} 'LockGuard'
// CHECK:   |   |-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   | |-<<<NULL>>>
// CHECK:   |   | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |   | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK:   |   | |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |   | |   | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue ','
// CHECK:   |   | |   |   |-CStyleCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'void' <ToVoid>
// CHECK:   |   | |   |   | `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' xvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |   | |   |   |   `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' (CXXTemporary 0x{{.+}})
// CHECK:   |   | |   |   |     `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK:   |   | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} 'v' 'int[3]'
// CHECK:   |   | |   `-typeDetails: LValueReferenceType 0x{{.+}} 'int (&)[3]'
// CHECK:   |   | |     `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |   | |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |   | | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'int *' cinit
// CHECK:   |   | |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |   | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |   | |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |   | |     `-typeDetails: DecayedType 0x{{.+}} 'int *' sugar
// CHECK:   |   | |       |-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |   | |       | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |       `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |   | |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |   | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __end1 'int *' cinit
// CHECK:   |   | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int *' '+'
// CHECK:   |   | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |   | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |   | |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'long' 3
// CHECK:   |   | |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |   | |     `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |   | |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:   |   | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |   | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |   | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__end1' 'int *'
// CHECK:   |   | |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int *' lvalue prefix '++'
// CHECK:   |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |   | |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |   | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} x 'int' cinit
// CHECK:   |   | |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |   | |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK:   |   | |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |   | |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |   | |   |-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |   `-attrDetails: UnusedAttr 0x{{.+}} <col:{{.*}}> maybe_unused
// CHECK:   |   | `-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:   |   |   `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} guard 'LockGuard':'P2718R0::LockGuard' callinit destroyed
// CHECK:   |   |     |-CXXConstructExpr 0x{{.+}} <col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK:   |   |     `-typeDetails: ElaboratedType 0x{{.+}} 'LockGuard' sugar
// CHECK:   |   |       `-typeDetails: RecordType 0x{{.+}} 'P2718R0::LockGuard'
// CHECK:   |   |         `-CXXRecord 0x{{.+}} 'LockGuard'
// CHECK:   |   `-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |     |-<<<NULL>>>
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK:   |     |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |     |   | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue ','
// CHECK:   |     |   |   |-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' xvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |   | `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |   |   `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK:   |     |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} 'v' 'int[3]'
// CHECK:   |     |   `-typeDetails: LValueReferenceType 0x{{.+}} 'int (&)[3]'
// CHECK:   |     |     `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |     |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'int *' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |     |     `-typeDetails: DecayedType 0x{{.+}} 'int *' sugar
// CHECK:   |     |       |-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |     |       | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |       `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __end1 'int *' cinit
// CHECK:   |     |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int *' '+'
// CHECK:   |     |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'long' 3
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |     |     `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |     |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:   |     | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__end1' 'int *'
// CHECK:   |     |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int *' lvalue prefix '++'
// CHECK:   |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} x 'int' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |     |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK:   |     |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     |   |-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |   `-attrDetails: UnusedAttr 0x{{.+}} <col:{{.*}}> maybe_unused
// CHECK:   |     `-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:   |       `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} guard 'LockGuard':'P2718R0::LockGuard' callinit destroyed
// CHECK:   |         |-CXXConstructExpr 0x{{.+}} <col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK:   |         `-typeDetails: ElaboratedType 0x{{.+}} 'LockGuard' sugar
// CHECK:   |           `-typeDetails: RecordType 0x{{.+}} 'P2718R0::LockGuard'
// CHECK:   |             `-CXXRecord 0x{{.+}} 'LockGuard'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used default_arg_fn 'int (&(const A &))[3]'
// CHECK:   | `-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} 'const A &' cinit
// CHECK:   |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' lvalue
// CHECK:   |   | `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' lvalue
// CHECK:   |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' <NoOp>
// CHECK:   |   |     `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |   |       `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' 'void ()'
// CHECK:   |   `-typeDetails: LValueReferenceType 0x{{.+}} 'const A &'
// CHECK:   |     `-qualTypeDetail: QualType 0x{{.+}} 'const A' const
// CHECK:   |       `-typeDetails: ElaboratedType 0x{{.+}} 'A' sugar
// CHECK:   |         `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   |           `-CXXRecord 0x{{.+}} 'A'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test4 'void ()'
// CHECK:   | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   `-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |     |-<<<NULL>>>
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK:   |     |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |     |   | `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |     |   |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int (&(*)(const A &))[3]' <FunctionToPointerDecay>
// CHECK:   |     |   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int (&(const A &))[3]' lvalue Function 0x{{.+}} 'default_arg_fn' 'int (&(const A &))[3]'
// CHECK:   |     |   |   `-CXXDefaultArgExpr 0x{{.+}} <<invalid sloc>> 'const A':'const P2718R0::A' lvalue has rewritten init
// CHECK:   |     |   |     `-MaterializeTemporaryExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |       `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' <NoOp>
// CHECK:   |     |   |         `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |           `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' 'void ()'
// CHECK:   |     |   `-typeDetails: LValueReferenceType 0x{{.+}} 'int (&)[3]'
// CHECK:   |     |     `-typeDetails: ParenType 0x{{.+}} 'int[3]' sugar
// CHECK:   |     |       `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'int *' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |     |     `-typeDetails: DecayedType 0x{{.+}} 'int *' sugar
// CHECK:   |     |       |-typeDetails: ParenType 0x{{.+}} 'int[3]' sugar
// CHECK:   |     |       | `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |     |       |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |       `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __end1 'int *' cinit
// CHECK:   |     |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int *' '+'
// CHECK:   |     |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'long' 3
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |     |     `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |     |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:   |     | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__end1' 'int *'
// CHECK:   |     |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int *' lvalue prefix '++'
// CHECK:   |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used e 'int' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |     |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK:   |     |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int' sugar
// CHECK:   |     |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     `-CallExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'void'
// CHECK:   |       |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'void (*)(int)' <FunctionToPointerDecay>
// CHECK:   |       | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'void (int)' lvalue Function 0x{{.+}} 'bar' 'void (int)'
// CHECK:   |       `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |         `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'e' 'int'
// CHECK:   |-CXXRecordDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} referenced struct DefaultA definition
// CHECK:   | |-DefinitionData empty standard_layout has_user_declared_ctor can_const_default_init
// CHECK:   | | |-DefaultConstructor exists non_trivial user_provided defaulted_is_constexpr
// CHECK:   | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK:   | | |-MoveConstructor
// CHECK:   | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:   | | |-MoveAssignment
// CHECK:   | | `-Destructor non_trivial user_declared
// CHECK:   | |-CXXRecordDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit referenced struct DefaultA
// CHECK:   | |-CXXConstructorDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used DefaultA 'void ()' implicit-inline
// CHECK:   | | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |-CXXDestructorDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used ~DefaultA 'void () noexcept' implicit-inline
// CHECK:   | | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | `-CXXConstructorDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit constexpr DefaultA 'void (const DefaultA &)' inline default trivial noexcept-unevaluated 0x{{.+}}
// CHECK:   |   `-ParmVarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 'const DefaultA &'
// CHECK:   |     `-typeDetails: LValueReferenceType 0x{{.+}} 'const DefaultA &'
// CHECK:   |       `-qualTypeDetail: QualType 0x{{.+}} 'const DefaultA' const
// CHECK:   |         `-typeDetails: ElaboratedType 0x{{.+}} 'DefaultA' sugar
// CHECK:   |           `-typeDetails: RecordType 0x{{.+}} 'P2718R0::DefaultA'
// CHECK:   |             `-CXXRecord 0x{{.+}} 'DefaultA'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} used foo 'A (const A &, const DefaultA &)'
// CHECK:   | |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} 'const A &'
// CHECK:   | | `-typeDetails: LValueReferenceType 0x{{.+}} 'const A &'
// CHECK:   | |   `-qualTypeDetail: QualType 0x{{.+}} 'const A' const
// CHECK:   | |     `-typeDetails: ElaboratedType 0x{{.+}} 'A' sugar
// CHECK:   | |       `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   | |         `-CXXRecord 0x{{.+}} 'A'
// CHECK:   | |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} Default 'const DefaultA &' cinit
// CHECK:   | | |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue
// CHECK:   | | | `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue
// CHECK:   | | |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK:   | | |     `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' (CXXTemporary 0x{{.+}})
// CHECK:   | | |       `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK:   | | `-typeDetails: LValueReferenceType 0x{{.+}} 'const DefaultA &'
// CHECK:   | |   `-qualTypeDetail: QualType 0x{{.+}} 'const DefaultA' const
// CHECK:   | |     `-typeDetails: ElaboratedType 0x{{.+}} 'DefaultA' sugar
// CHECK:   | |       `-typeDetails: RecordType 0x{{.+}} 'P2718R0::DefaultA'
// CHECK:   | |         `-CXXRecord 0x{{.+}} 'DefaultA'
// CHECK:   | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   `-ReturnStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:   |     `-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:   |       `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |         `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' 'void ()'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test5 'void ()'
// CHECK:   | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   `-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |     |-<<<NULL>>>
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK:   |     |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |     |   | `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |     |   |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int (&(*)(const A &))[3]' <FunctionToPointerDecay>
// CHECK:   |     |   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int (&(const A &))[3]' lvalue Function 0x{{.+}} 'default_arg_fn' 'int (&(const A &))[3]'
// CHECK:   |     |   |   `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |     `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' <NoOp>
// CHECK:   |     |   |       `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |         `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:   |     |   |           |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'A (*)(const A &, const DefaultA &)' <FunctionToPointerDecay>
// CHECK:   |     |   |           | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'A (const A &, const DefaultA &)' lvalue Function 0x{{.+}} 'foo' 'A (const A &, const DefaultA &)'
// CHECK:   |     |   |           |-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |           | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' <NoOp>
// CHECK:   |     |   |           |   `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |           |     `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:   |     |   |           |       |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'A (*)(const A &, const DefaultA &)' <FunctionToPointerDecay>
// CHECK:   |     |   |           |       | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'A (const A &, const DefaultA &)' lvalue Function 0x{{.+}} 'foo' 'A (const A &, const DefaultA &)'
// CHECK:   |     |   |           |       |-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |           |       | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' <NoOp>
// CHECK:   |     |   |           |       |   `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |           |       |     `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:   |     |   |           |       |       |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'A (*)(const A &, const DefaultA &)' <FunctionToPointerDecay>
// CHECK:   |     |   |           |       |       | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'A (const A &, const DefaultA &)' lvalue Function 0x{{.+}} 'foo' 'A (const A &, const DefaultA &)'
// CHECK:   |     |   |           |       |       |-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |           |       |       | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' <NoOp>
// CHECK:   |     |   |           |       |       |   `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |           |       |       |     `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' 'void ()'
// CHECK:   |     |   |           |       |       `-CXXDefaultArgExpr 0x{{.+}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK:   |     |   |           |       |         `-MaterializeTemporaryExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |           |       |           `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK:   |     |   |           |       |             `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |           |       |               `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK:   |     |   |           |       `-CXXDefaultArgExpr 0x{{.+}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK:   |     |   |           |         `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |           |           `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK:   |     |   |           |             `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |           |               `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK:   |     |   |           `-CXXDefaultArgExpr 0x{{.+}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK:   |     |   |             `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |               `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK:   |     |   |                 `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |                   `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK:   |     |   `-typeDetails: LValueReferenceType 0x{{.+}} 'int (&)[3]'
// CHECK:   |     |     `-typeDetails: ParenType 0x{{.+}} 'int[3]' sugar
// CHECK:   |     |       `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'int *' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |     |     `-typeDetails: DecayedType 0x{{.+}} 'int *' sugar
// CHECK:   |     |       |-typeDetails: ParenType 0x{{.+}} 'int[3]' sugar
// CHECK:   |     |       | `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |     |       |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |       `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __end1 'int *' cinit
// CHECK:   |     |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int *' '+'
// CHECK:   |     |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'long' 3
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |     |     `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |     |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:   |     | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__end1' 'int *'
// CHECK:   |     |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int *' lvalue prefix '++'
// CHECK:   |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used e 'int' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |     |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK:   |     |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int' sugar
// CHECK:   |     |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     `-CallExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'void'
// CHECK:   |       |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'void (*)(int)' <FunctionToPointerDecay>
// CHECK:   |       | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'void (int)' lvalue Function 0x{{.+}} 'bar' 'void (int)'
// CHECK:   |       `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |         `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'e' 'int'
// CHECK:   |-CXXRecordDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} referenced struct C definition
// CHECK:   | |-DefinitionData standard_layout has_user_declared_ctor can_const_default_init
// CHECK:   | | |-DefaultConstructor exists non_trivial user_provided defaulted_is_constexpr
// CHECK:   | | |-CopyConstructor simple trivial has_const_param needs_overload_resolution implicit_has_const_param
// CHECK:   | | |-MoveConstructor exists simple trivial needs_overload_resolution
// CHECK:   | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:   | | |-MoveAssignment exists simple trivial needs_overload_resolution
// CHECK:   | | `-Destructor simple non_trivial constexpr needs_overload_resolution
// CHECK:   | |-public 'A':'P2718R0::A'
// CHECK:   | |-CXXRecordDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit referenced struct C
// CHECK:   | |-CXXConstructorDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used C 'void ()' implicit-inline
// CHECK:   | | |-CXXCtorInitializer 'A':'P2718R0::A'
// CHECK:   | | | `-CXXConstructExpr 0x{{.+}} <col:{{.*}}> 'A':'P2718R0::A' 'void ()'
// CHECK:   | | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |-CXXConstructorDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used C 'void (int, const C &, const DefaultA &)' implicit-inline
// CHECK:   | | |-ParmVarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 'int'
// CHECK:   | | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   | | |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} 'const C &'
// CHECK:   | | | `-typeDetails: LValueReferenceType 0x{{.+}} 'const C &'
// CHECK:   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const C' const
// CHECK:   | | |     `-typeDetails: ElaboratedType 0x{{.+}} 'C' sugar
// CHECK:   | | |       `-typeDetails: RecordType 0x{{.+}} 'P2718R0::C'
// CHECK:   | | |         `-CXXRecord 0x{{.+}} 'C'
// CHECK:   | | |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} 'const DefaultA &' cinit
// CHECK:   | | | |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue
// CHECK:   | | | | `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue
// CHECK:   | | | |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK:   | | | |     `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' (CXXTemporary 0x{{.+}})
// CHECK:   | | | |       `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK:   | | | `-typeDetails: LValueReferenceType 0x{{.+}} 'const DefaultA &'
// CHECK:   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const DefaultA' const
// CHECK:   | | |     `-typeDetails: ElaboratedType 0x{{.+}} 'DefaultA' sugar
// CHECK:   | | |       `-typeDetails: RecordType 0x{{.+}} 'P2718R0::DefaultA'
// CHECK:   | | |         `-CXXRecord 0x{{.+}} 'DefaultA'
// CHECK:   | | |-CXXCtorInitializer 'A':'P2718R0::A'
// CHECK:   | | | `-CXXConstructExpr 0x{{.+}} <col:{{.*}}> 'A':'P2718R0::A' 'void ()'
// CHECK:   | | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |-CXXConstructorDecl 0x{{.+}} <line:{{.*}}:{{.*}}> col:{{.*}} implicit constexpr C 'void (const C &)' inline default trivial noexcept-unevaluated 0x{{.+}}
// CHECK:   | | `-ParmVarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 'const C &'
// CHECK:   | |   `-typeDetails: LValueReferenceType 0x{{.+}} 'const C &'
// CHECK:   | |     `-qualTypeDetail: QualType 0x{{.+}} 'const C' const
// CHECK:   | |       `-typeDetails: ElaboratedType 0x{{.+}} 'C' sugar
// CHECK:   | |         `-typeDetails: RecordType 0x{{.+}} 'P2718R0::C'
// CHECK:   | |           `-CXXRecord 0x{{.+}} 'C'
// CHECK:   | |-CXXConstructorDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit constexpr C 'void (C &&)' inline default trivial noexcept-unevaluated 0x{{.+}}
// CHECK:   | | `-ParmVarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 'C &&'
// CHECK:   | |   `-typeDetails: RValueReferenceType 0x{{.+}} 'C &&'
// CHECK:   | |     `-typeDetails: ElaboratedType 0x{{.+}} 'C' sugar
// CHECK:   | |       `-typeDetails: RecordType 0x{{.+}} 'P2718R0::C'
// CHECK:   | |         `-CXXRecord 0x{{.+}} 'C'
// CHECK:   | |-CXXMethodDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit constexpr operator= 'C &(C &&)' inline default trivial noexcept-unevaluated 0x{{.+}}
// CHECK:   | | `-ParmVarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} 'C &&'
// CHECK:   | |   `-typeDetails: RValueReferenceType 0x{{.+}} 'C &&'
// CHECK:   | |     `-typeDetails: ElaboratedType 0x{{.+}} 'C' sugar
// CHECK:   | |       `-typeDetails: RecordType 0x{{.+}} 'P2718R0::C'
// CHECK:   | |         `-CXXRecord 0x{{.+}} 'C'
// CHECK:   | `-CXXDestructorDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used constexpr ~C 'void () noexcept' inline default
// CHECK:   |   `-CompoundStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test6 'void ()'
// CHECK:   | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   `-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |     |-<<<NULL>>>
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'C &&' cinit
// CHECK:   |     |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'C':'P2718R0::C' xvalue
// CHECK:   |     |   | `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'C':'P2718R0::C' xvalue extended by Var 0x{{.+}} '__range1' 'C &&'
// CHECK:   |     |   |   `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'C':'P2718R0::C' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |     `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'C':'P2718R0::C' 'void (int, const C &, const DefaultA &)'
// CHECK:   |     |   |       |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:   |     |   |       |-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const C':'const P2718R0::C' lvalue extended by Var 0x{{.+}} '__range1' 'C &&'
// CHECK:   |     |   |       | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const C':'const P2718R0::C' <NoOp>
// CHECK:   |     |   |       |   `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'C':'P2718R0::C' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |       |     `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'C':'P2718R0::C' 'void (int, const C &, const DefaultA &)'
// CHECK:   |     |   |       |       |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:   |     |   |       |       |-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const C':'const P2718R0::C' lvalue extended by Var 0x{{.+}} '__range1' 'C &&'
// CHECK:   |     |   |       |       | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const C':'const P2718R0::C' <NoOp>
// CHECK:   |     |   |       |       |   `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'C':'P2718R0::C' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |       |       |     `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'C':'P2718R0::C' 'void (int, const C &, const DefaultA &)'
// CHECK:   |     |   |       |       |       |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:   |     |   |       |       |       |-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const C':'const P2718R0::C' lvalue extended by Var 0x{{.+}} '__range1' 'C &&'
// CHECK:   |     |   |       |       |       | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const C':'const P2718R0::C' <NoOp>
// CHECK:   |     |   |       |       |       |   `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'C':'P2718R0::C' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |       |       |       |     `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'C':'P2718R0::C' 'void ()'
// CHECK:   |     |   |       |       |       `-CXXDefaultArgExpr 0x{{.+}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK:   |     |   |       |       |         `-MaterializeTemporaryExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var 0x{{.+}} '__range1' 'C &&'
// CHECK:   |     |   |       |       |           `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK:   |     |   |       |       |             `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |       |       |               `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK:   |     |   |       |       `-CXXDefaultArgExpr 0x{{.+}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK:   |     |   |       |         `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var 0x{{.+}} '__range1' 'C &&'
// CHECK:   |     |   |       |           `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK:   |     |   |       |             `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |       |               `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK:   |     |   |       `-CXXDefaultArgExpr 0x{{.+}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK:   |     |   |         `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var 0x{{.+}} '__range1' 'C &&'
// CHECK:   |     |   |           `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK:   |     |   |             `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |               `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK:   |     |   `-typeDetails: RValueReferenceType 0x{{.+}} 'C &&'
// CHECK:   |     |     `-typeDetails: AutoType 0x{{.+}} 'C' sugar
// CHECK:   |     |       `-typeDetails: ElaboratedType 0x{{.+}} 'C' sugar
// CHECK:   |     |         `-typeDetails: RecordType 0x{{.+}} 'P2718R0::C'
// CHECK:   |     |           `-CXXRecord 0x{{.+}} 'C'
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'const int *' cinit
// CHECK:   |     |   |-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}> 'const int *'
// CHECK:   |     |   | `-MemberExpr 0x{{.+}} <col:{{.*}}> '<bound member function type>' .begin 0x{{.+}}
// CHECK:   |     |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::A' lvalue <UncheckedDerivedToBase (A)>
// CHECK:   |     |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'C':'P2718R0::C' lvalue Var 0x{{.+}} '__range1' 'C &&'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'const int *' sugar
// CHECK:   |     |     `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:   |     |       `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __end1 'const int *' cinit
// CHECK:   |     |   |-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}> 'const int *'
// CHECK:   |     |   | `-MemberExpr 0x{{.+}} <col:{{.*}}> '<bound member function type>' .end 0x{{.+}}
// CHECK:   |     |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::A' lvalue <UncheckedDerivedToBase (A)>
// CHECK:   |     |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'C':'P2718R0::C' lvalue Var 0x{{.+}} '__range1' 'C &&'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'const int *' sugar
// CHECK:   |     |     `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:   |     |       `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:   |     | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <LValueToRValue>
// CHECK:   |     | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__begin1' 'const int *'
// CHECK:   |     | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <LValueToRValue>
// CHECK:   |     |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__end1' 'const int *'
// CHECK:   |     |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'const int *' lvalue prefix '++'
// CHECK:   |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__begin1' 'const int *'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used e 'int' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |     |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'const int' lvalue prefix '*' cannot overflow
// CHECK:   |     |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <LValueToRValue>
// CHECK:   |     |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__begin1' 'const int *'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int' sugar
// CHECK:   |     |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     `-CallExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'void'
// CHECK:   |       |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'void (*)(int)' <FunctionToPointerDecay>
// CHECK:   |       | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'void (int)' lvalue Function 0x{{.+}} 'bar' 'void (int)'
// CHECK:   |       `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |         `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'e' 'int'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test7 'void ()'
// CHECK:   | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   `-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |     |-<<<NULL>>>
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'A &&' cinit
// CHECK:   |     |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' xvalue
// CHECK:   |     |   | `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' xvalue extended by Var 0x{{.+}} '__range1' 'A &&'
// CHECK:   |     |   |   `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |     `-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:   |     |   |       `-MemberExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<bound member function type>' .g 0x{{.+}}
// CHECK:   |     |   |         `-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' lvalue
// CHECK:   |     |   |           `-MemberExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<bound member function type>' .r 0x{{.+}}
// CHECK:   |     |   |             `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' xvalue extended by Var 0x{{.+}} '__range1' 'A &&'
// CHECK:   |     |   |               `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |                 `-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:   |     |   |                   `-MemberExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<bound member function type>' .g 0x{{.+}}
// CHECK:   |     |   |                     `-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' lvalue
// CHECK:   |     |   |                       `-MemberExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<bound member function type>' .r 0x{{.+}}
// CHECK:   |     |   |                         `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' xvalue extended by Var 0x{{.+}} '__range1' 'A &&'
// CHECK:   |     |   |                           `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |                             `-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:   |     |   |                               `-MemberExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<bound member function type>' .g 0x{{.+}}
// CHECK:   |     |   |                                 `-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' lvalue
// CHECK:   |     |   |                                   `-MemberExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<bound member function type>' .r 0x{{.+}}
// CHECK:   |     |   |                                     `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' xvalue extended by Var 0x{{.+}} '__range1' 'A &&'
// CHECK:   |     |   |                                       `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |                                         `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:   |     |   |                                           `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'A (*)()' <FunctionToPointerDecay>
// CHECK:   |     |   |                                             `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'A ()' lvalue Function 0x{{.+}} 'g' 'A ()'
// CHECK:   |     |   `-typeDetails: RValueReferenceType 0x{{.+}} 'A &&'
// CHECK:   |     |     `-typeDetails: AutoType 0x{{.+}} 'A' sugar
// CHECK:   |     |       `-typeDetails: ElaboratedType 0x{{.+}} 'A' sugar
// CHECK:   |     |         `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   |     |           `-CXXRecord 0x{{.+}} 'A'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'const int *' cinit
// CHECK:   |     |   |-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}> 'const int *'
// CHECK:   |     |   | `-MemberExpr 0x{{.+}} <col:{{.*}}> '<bound member function type>' .begin 0x{{.+}}
// CHECK:   |     |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::A' lvalue <NoOp>
// CHECK:   |     |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'A':'P2718R0::A' lvalue Var 0x{{.+}} '__range1' 'A &&'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'const int *' sugar
// CHECK:   |     |     `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:   |     |       `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __end1 'const int *' cinit
// CHECK:   |     |   |-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}> 'const int *'
// CHECK:   |     |   | `-MemberExpr 0x{{.+}} <col:{{.*}}> '<bound member function type>' .end 0x{{.+}}
// CHECK:   |     |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::A' lvalue <NoOp>
// CHECK:   |     |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'A':'P2718R0::A' lvalue Var 0x{{.+}} '__range1' 'A &&'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'const int *' sugar
// CHECK:   |     |     `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:   |     |       `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:   |     | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <LValueToRValue>
// CHECK:   |     | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__begin1' 'const int *'
// CHECK:   |     | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <LValueToRValue>
// CHECK:   |     |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__end1' 'const int *'
// CHECK:   |     |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'const int *' lvalue prefix '++'
// CHECK:   |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__begin1' 'const int *'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used e 'int' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |     |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'const int' lvalue prefix '*' cannot overflow
// CHECK:   |     |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <LValueToRValue>
// CHECK:   |     |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__begin1' 'const int *'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int' sugar
// CHECK:   |     |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     `-CallExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'void'
// CHECK:   |       |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'void (*)(int)' <FunctionToPointerDecay>
// CHECK:   |       | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'void (int)' lvalue Function 0x{{.+}} 'bar' 'void (int)'
// CHECK:   |       `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |         `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'e' 'int'
// CHECK:   |-FunctionTemplateDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} dg
// CHECK:   | |-TemplateTypeParmDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} referenced typename depth 0 index 0 T
// CHECK:   | |-FunctionDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} dg 'T ()'
// CHECK:   | | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |   `-ReturnStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |     `-CXXUnresolvedConstructExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'T' 'T'
// CHECK:   | `-FunctionDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used dg 'P2718R0::A ()' implicit_instantiation
// CHECK:   |   |-TemplateArgument type 'P2718R0::A'
// CHECK:   |   | `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   |   |   `-CXXRecord 0x{{.+}} 'A'
// CHECK:   |   `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |     `-ReturnStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |       `-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A'
// CHECK:   |         `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |           `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A' 'void ()'
// CHECK:   |-FunctionTemplateDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} df1
// CHECK:   | |-TemplateTypeParmDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} referenced typename depth 0 index 0 T
// CHECK:   | |-FunctionDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} df1 'const T &(const T &)'
// CHECK:   | | |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} referenced t 'const T &'
// CHECK:   | | | `-typeDetails: LValueReferenceType 0x{{.+}} 'const T &' dependent
// CHECK:   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const T' const
// CHECK:   | | |     `-typeDetails: TemplateTypeParmType 0x{{.+}} 'T' dependent depth 0 index 0
// CHECK:   | | |       `-TemplateTypeParm 0x{{.+}} 'T'
// CHECK:   | | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |   `-ReturnStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const T' lvalue ParmVar 0x{{.+}} 't' 'const T &'
// CHECK:   | |-FunctionDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used df1 'const P2718R0::A &(const P2718R0::A &)' implicit_instantiation
// CHECK:   | | |-TemplateArgument type 'P2718R0::A'
// CHECK:   | | | `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   | | |   `-CXXRecord 0x{{.+}} 'A'
// CHECK:   | | |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used t 'const P2718R0::A &'
// CHECK:   | | | `-typeDetails: LValueReferenceType 0x{{.+}} 'const P2718R0::A &'
// CHECK:   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const P2718R0::A' const
// CHECK:   | | |     `-typeDetails: SubstTemplateTypeParmType 0x{{.+}} 'P2718R0::A' sugar typename depth 0 index 0 T
// CHECK:   | | |       |-FunctionTemplate 0x{{.+}} 'df1'
// CHECK:   | | |       `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   | | |         `-CXXRecord 0x{{.+}} 'A'
// CHECK:   | | `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |   `-ReturnStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   | |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::A' lvalue ParmVar 0x{{.+}} 't' 'const P2718R0::A &'
// CHECK:   | `-FunctionDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used df1 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' implicit_instantiation
// CHECK:   |   |-TemplateArgument type 'P2718R0::LockGuard'
// CHECK:   |   | `-typeDetails: RecordType 0x{{.+}} 'P2718R0::LockGuard'
// CHECK:   |   |   `-CXXRecord 0x{{.+}} 'LockGuard'
// CHECK:   |   |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used t 'const P2718R0::LockGuard &'
// CHECK:   |   | `-typeDetails: LValueReferenceType 0x{{.+}} 'const P2718R0::LockGuard &'
// CHECK:   |   |   `-qualTypeDetail: QualType 0x{{.+}} 'const P2718R0::LockGuard' const
// CHECK:   |   |     `-typeDetails: SubstTemplateTypeParmType 0x{{.+}} 'P2718R0::LockGuard' sugar typename depth 0 index 0 T
// CHECK:   |   |       |-FunctionTemplate 0x{{.+}} 'df1'
// CHECK:   |   |       `-typeDetails: RecordType 0x{{.+}} 'P2718R0::LockGuard'
// CHECK:   |   |         `-CXXRecord 0x{{.+}} 'LockGuard'
// CHECK:   |   `-CompoundStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |     `-ReturnStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |       `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::LockGuard' lvalue ParmVar 0x{{.+}} 't' 'const P2718R0::LockGuard &'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test8 'void ()'
// CHECK:   | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:   |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used sum 'int' cinit
// CHECK:   |   |   |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 0
// CHECK:   |   |   |-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   |   `-attrDetails: UnusedAttr 0x{{.+}} <col:{{.*}}> maybe_unused
// CHECK:   |   `-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |     |-<<<NULL>>>
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'const P2718R0::A &' cinit
// CHECK:   |     |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const P2718R0::A' lvalue
// CHECK:   |     |   | `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const P2718R0::A' lvalue
// CHECK:   |     |   |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::A &(*)(const P2718R0::A &)' <FunctionToPointerDecay>
// CHECK:   |     |   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::A &(const P2718R0::A &)' lvalue Function 0x{{.+}} 'df1' 'const P2718R0::A &(const P2718R0::A &)' (FunctionTemplate 0x{{.+}} 'df1')
// CHECK:   |     |   |   `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const P2718R0::A' lvalue extended by Var 0x{{.+}} '__range1' 'const P2718R0::A &'
// CHECK:   |     |   |     `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const P2718R0::A' <NoOp>
// CHECK:   |     |   |       `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |         `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A'
// CHECK:   |     |   |           `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A (*)()' <FunctionToPointerDecay>
// CHECK:   |     |   |             `-DeclRefExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A ()' lvalue Function 0x{{.+}} 'dg' 'P2718R0::A ()' (FunctionTemplate 0x{{.+}} 'dg')
// CHECK:   |     |   `-typeDetails: LValueReferenceType 0x{{.+}} 'const P2718R0::A &'
// CHECK:   |     |     `-qualTypeDetail: QualType 0x{{.+}} 'const P2718R0::A' const
// CHECK:   |     |       `-typeDetails: SubstTemplateTypeParmType 0x{{.+}} 'P2718R0::A' sugar typename depth 0 index 0 T
// CHECK:   |     |         |-FunctionTemplate 0x{{.+}} 'df1'
// CHECK:   |     |         `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   |     |           `-CXXRecord 0x{{.+}} 'A'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'const int *' cinit
// CHECK:   |     |   |-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}> 'const int *'
// CHECK:   |     |   | `-MemberExpr 0x{{.+}} <col:{{.*}}> '<bound member function type>' .begin 0x{{.+}}
// CHECK:   |     |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::A' lvalue Var 0x{{.+}} '__range1' 'const P2718R0::A &'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'const int *' sugar
// CHECK:   |     |     `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:   |     |       `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __end1 'const int *' cinit
// CHECK:   |     |   |-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}> 'const int *'
// CHECK:   |     |   | `-MemberExpr 0x{{.+}} <col:{{.*}}> '<bound member function type>' .end 0x{{.+}}
// CHECK:   |     |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::A' lvalue Var 0x{{.+}} '__range1' 'const P2718R0::A &'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'const int *' sugar
// CHECK:   |     |     `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:   |     |       `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:   |     | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <LValueToRValue>
// CHECK:   |     | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__begin1' 'const int *'
// CHECK:   |     | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <LValueToRValue>
// CHECK:   |     |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__end1' 'const int *'
// CHECK:   |     |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'const int *' lvalue prefix '++'
// CHECK:   |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__begin1' 'const int *'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used e 'int' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |     |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'const int' lvalue prefix '*' cannot overflow
// CHECK:   |     |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <LValueToRValue>
// CHECK:   |     |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__begin1' 'const int *'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int' sugar
// CHECK:   |     |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     `-CompoundAssignOperator 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'int' lvalue '+=' ComputeLHSTy='int' ComputeResultTy='int'
// CHECK:   |       |-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'sum' 'int'
// CHECK:   |       `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |         `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'e' 'int'
// CHECK:   |-FunctionTemplateDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} df2
// CHECK:   | |-TemplateTypeParmDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} referenced typename depth 0 index 0 T
// CHECK:   | |-FunctionDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} df2 'int (&(const T *))[3]'
// CHECK:   | | `-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} 'const T *'
// CHECK:   | |   `-typeDetails: PointerType 0x{{.+}} 'const T *' dependent
// CHECK:   | |     `-qualTypeDetail: QualType 0x{{.+}} 'const T' const
// CHECK:   | |       `-typeDetails: TemplateTypeParmType 0x{{.+}} 'T' dependent depth 0 index 0
// CHECK:   | |         `-TemplateTypeParm 0x{{.+}} 'T'
// CHECK:   | `-FunctionDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used df2 'int (&(const P2718R0::A *))[3]' implicit_instantiation
// CHECK:   |   |-TemplateArgument type 'P2718R0::A'
// CHECK:   |   | `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   |   |   `-CXXRecord 0x{{.+}} 'A'
// CHECK:   |   `-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} 'const P2718R0::A *'
// CHECK:   |     `-typeDetails: PointerType 0x{{.+}} 'const P2718R0::A *'
// CHECK:   |       `-qualTypeDetail: QualType 0x{{.+}} 'const P2718R0::A' const
// CHECK:   |         `-typeDetails: SubstTemplateTypeParmType 0x{{.+}} 'P2718R0::A' sugar typename depth 0 index 0 T
// CHECK:   |           |-FunctionTemplate 0x{{.+}} 'df2'
// CHECK:   |           `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   |             `-CXXRecord 0x{{.+}} 'A'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} used dg2 'const A *(const A &)'
// CHECK:   | `-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} 'const A &'
// CHECK:   |   `-typeDetails: LValueReferenceType 0x{{.+}} 'const A &'
// CHECK:   |     `-qualTypeDetail: QualType 0x{{.+}} 'const A' const
// CHECK:   |       `-typeDetails: ElaboratedType 0x{{.+}} 'A' sugar
// CHECK:   |         `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   |           `-CXXRecord 0x{{.+}} 'A'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test9 'void ()'
// CHECK:   | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   `-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |     |-<<<NULL>>>
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK:   |     |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |     |   | `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |     |   |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int (&(*)(const P2718R0::A *))[3]' <FunctionToPointerDecay>
// CHECK:   |     |   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int (&(const P2718R0::A *))[3]' lvalue Function 0x{{.+}} 'df2' 'int (&(const P2718R0::A *))[3]' (FunctionTemplate 0x{{.+}} 'df2')
// CHECK:   |     |   |   `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A *'
// CHECK:   |     |   |     |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const A *(*)(const A &)' <FunctionToPointerDecay>
// CHECK:   |     |   |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const A *(const A &)' lvalue Function 0x{{.+}} 'dg2' 'const A *(const A &)'
// CHECK:   |     |   |     `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' lvalue <DerivedToBase (A)>
// CHECK:   |     |   |       `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const B':'const P2718R0::B' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |         `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const B':'const P2718R0::B' <NoOp>
// CHECK:   |     |   |           `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'B':'P2718R0::B' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |             `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'B':'P2718R0::B' 'void () noexcept(false)' zeroing
// CHECK:   |     |   `-typeDetails: LValueReferenceType 0x{{.+}} 'int (&)[3]'
// CHECK:   |     |     `-typeDetails: ParenType 0x{{.+}} 'int[3]' sugar
// CHECK:   |     |       `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'int *' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |     |     `-typeDetails: DecayedType 0x{{.+}} 'int *' sugar
// CHECK:   |     |       |-typeDetails: ParenType 0x{{.+}} 'int[3]' sugar
// CHECK:   |     |       | `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |     |       |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |       `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __end1 'int *' cinit
// CHECK:   |     |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int *' '+'
// CHECK:   |     |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'long' 3
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |     |     `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |     |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:   |     | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__end1' 'int *'
// CHECK:   |     |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int *' lvalue prefix '++'
// CHECK:   |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used e 'int' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |     |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK:   |     |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int' sugar
// CHECK:   |     |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     `-CallExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'void'
// CHECK:   |       |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'void (*)(int)' <FunctionToPointerDecay>
// CHECK:   |       | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'void (int)' lvalue Function 0x{{.+}} 'bar' 'void (int)'
// CHECK:   |       `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |         `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'e' 'int'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test10 'void ()'
// CHECK:   | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:   |   | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used v 'int[3]' cinit
// CHECK:   |   |   |-InitListExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]'
// CHECK:   |   |   | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 42
// CHECK:   |   |   | |-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 17
// CHECK:   |   |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'int' 13
// CHECK:   |   |   `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |   |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   |-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   | |-<<<NULL>>>
// CHECK:   |   | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |   | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK:   |   | |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |   | |   | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue ','
// CHECK:   |   | |   |   |-CXXStaticCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'void' static_cast<void> <ToVoid>
// CHECK:   |   | |   |   | `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const P2718R0::LockGuard' lvalue
// CHECK:   |   | |   |   |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::LockGuard &(*)(const P2718R0::LockGuard &)' <FunctionToPointerDecay>
// CHECK:   |   | |   |   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' lvalue Function 0x{{.+}} 'df1' 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' (FunctionTemplate 0x{{.+}} 'df1')
// CHECK:   |   | |   |   |   `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const LockGuard':'const P2718R0::LockGuard' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |   | |   |   |     `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const LockGuard':'const P2718R0::LockGuard' <NoOp>
// CHECK:   |   | |   |   |       `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' (CXXTemporary 0x{{.+}})
// CHECK:   |   | |   |   |         `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK:   |   | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} 'v' 'int[3]'
// CHECK:   |   | |   `-typeDetails: LValueReferenceType 0x{{.+}} 'int (&)[3]'
// CHECK:   |   | |     `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |   | |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |   | | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'int *' cinit
// CHECK:   |   | |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |   | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |   | |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |   | |     `-typeDetails: DecayedType 0x{{.+}} 'int *' sugar
// CHECK:   |   | |       |-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |   | |       | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |       `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |   | |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |   | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __end1 'int *' cinit
// CHECK:   |   | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int *' '+'
// CHECK:   |   | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |   | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |   | |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'long' 3
// CHECK:   |   | |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |   | |     `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |   | |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:   |   | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |   | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |   | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__end1' 'int *'
// CHECK:   |   | |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int *' lvalue prefix '++'
// CHECK:   |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |   | |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |   | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} x 'int' cinit
// CHECK:   |   | |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |   | |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK:   |   | |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |   | |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |   | |   |-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |   `-attrDetails: UnusedAttr 0x{{.+}} <col:{{.*}}> maybe_unused
// CHECK:   |   | `-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:   |   |   `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} guard 'LockGuard':'P2718R0::LockGuard' callinit destroyed
// CHECK:   |   |     |-CXXConstructExpr 0x{{.+}} <col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK:   |   |     `-typeDetails: ElaboratedType 0x{{.+}} 'LockGuard' sugar
// CHECK:   |   |       `-typeDetails: RecordType 0x{{.+}} 'P2718R0::LockGuard'
// CHECK:   |   |         `-CXXRecord 0x{{.+}} 'LockGuard'
// CHECK:   |   |-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   | |-<<<NULL>>>
// CHECK:   |   | |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |   | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK:   |   | |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |   | |   | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue ','
// CHECK:   |   | |   |   |-CStyleCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'void' <ToVoid>
// CHECK:   |   | |   |   | `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const P2718R0::LockGuard' lvalue
// CHECK:   |   | |   |   |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::LockGuard &(*)(const P2718R0::LockGuard &)' <FunctionToPointerDecay>
// CHECK:   |   | |   |   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' lvalue Function 0x{{.+}} 'df1' 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' (FunctionTemplate 0x{{.+}} 'df1')
// CHECK:   |   | |   |   |   `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const LockGuard':'const P2718R0::LockGuard' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |   | |   |   |     `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const LockGuard':'const P2718R0::LockGuard' <NoOp>
// CHECK:   |   | |   |   |       `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' (CXXTemporary 0x{{.+}})
// CHECK:   |   | |   |   |         `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK:   |   | |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} 'v' 'int[3]'
// CHECK:   |   | |   `-typeDetails: LValueReferenceType 0x{{.+}} 'int (&)[3]'
// CHECK:   |   | |     `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |   | |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |   | | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'int *' cinit
// CHECK:   |   | |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |   | |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |   | |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |   | |     `-typeDetails: DecayedType 0x{{.+}} 'int *' sugar
// CHECK:   |   | |       |-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |   | |       | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |       `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |   | |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |   | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __end1 'int *' cinit
// CHECK:   |   | |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int *' '+'
// CHECK:   |   | |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |   | |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |   | |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'long' 3
// CHECK:   |   | |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |   | |     `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |   | |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:   |   | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |   | | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |   | |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__end1' 'int *'
// CHECK:   |   | |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int *' lvalue prefix '++'
// CHECK:   |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |   | |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |   | | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} x 'int' cinit
// CHECK:   |   | |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |   | |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK:   |   | |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |   | |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |   | |   |-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |   | |   `-attrDetails: UnusedAttr 0x{{.+}} <col:{{.*}}> maybe_unused
// CHECK:   |   | `-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:   |   |   `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} guard 'LockGuard':'P2718R0::LockGuard' callinit destroyed
// CHECK:   |   |     |-CXXConstructExpr 0x{{.+}} <col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK:   |   |     `-typeDetails: ElaboratedType 0x{{.+}} 'LockGuard' sugar
// CHECK:   |   |       `-typeDetails: RecordType 0x{{.+}} 'P2718R0::LockGuard'
// CHECK:   |   |         `-CXXRecord 0x{{.+}} 'LockGuard'
// CHECK:   |   `-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |     |-<<<NULL>>>
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK:   |     |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |     |   | `-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue ','
// CHECK:   |     |   |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const P2718R0::LockGuard' lvalue ','
// CHECK:   |     |   |   | |-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const P2718R0::LockGuard' lvalue
// CHECK:   |     |   |   | | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::LockGuard &(*)(const P2718R0::LockGuard &)' <FunctionToPointerDecay>
// CHECK:   |     |   |   | | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' lvalue Function 0x{{.+}} 'df1' 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' (FunctionTemplate 0x{{.+}} 'df1')
// CHECK:   |     |   |   | | `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const LockGuard':'const P2718R0::LockGuard' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |   | |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const LockGuard':'const P2718R0::LockGuard' <NoOp>
// CHECK:   |     |   |   | |     `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |   | |       `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK:   |     |   |   | `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const P2718R0::LockGuard' lvalue
// CHECK:   |     |   |   |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::LockGuard &(*)(const P2718R0::LockGuard &)' <FunctionToPointerDecay>
// CHECK:   |     |   |   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' lvalue Function 0x{{.+}} 'df1' 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' (FunctionTemplate 0x{{.+}} 'df1')
// CHECK:   |     |   |   |   `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const LockGuard':'const P2718R0::LockGuard' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |   |     `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const LockGuard':'const P2718R0::LockGuard' <NoOp>
// CHECK:   |     |   |   |       `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |   |         `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK:   |     |   |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} 'v' 'int[3]'
// CHECK:   |     |   `-typeDetails: LValueReferenceType 0x{{.+}} 'int (&)[3]'
// CHECK:   |     |     `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |     |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'int *' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |     |     `-typeDetails: DecayedType 0x{{.+}} 'int *' sugar
// CHECK:   |     |       |-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |     |       | `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |       `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __end1 'int *' cinit
// CHECK:   |     |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int *' '+'
// CHECK:   |     |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'long' 3
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |     |     `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |     |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:   |     | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__end1' 'int *'
// CHECK:   |     |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int *' lvalue prefix '++'
// CHECK:   |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} x 'int' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |     |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK:   |     |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     |   |-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |   `-attrDetails: UnusedAttr 0x{{.+}} <col:{{.*}}> maybe_unused
// CHECK:   |     `-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}>
// CHECK:   |       `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} guard 'LockGuard':'P2718R0::LockGuard' callinit destroyed
// CHECK:   |         |-CXXConstructExpr 0x{{.+}} <col:{{.*}}> 'LockGuard':'P2718R0::LockGuard' 'void ()'
// CHECK:   |         `-typeDetails: ElaboratedType 0x{{.+}} 'LockGuard' sugar
// CHECK:   |           `-typeDetails: RecordType 0x{{.+}} 'P2718R0::LockGuard'
// CHECK:   |             `-CXXRecord 0x{{.+}} 'LockGuard'
// CHECK:   |-FunctionTemplateDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} default_arg_fn2
// CHECK:   | |-TemplateTypeParmDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} referenced typename depth 0 index 0 T
// CHECK:   | |-FunctionDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} default_arg_fn2 'int (&(const T &))[3]'
// CHECK:   | | `-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} 'const T &' cinit
// CHECK:   | |   |-CXXUnresolvedConstructExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'T' 'T'
// CHECK:   | |   `-typeDetails: LValueReferenceType 0x{{.+}} 'const T &' dependent
// CHECK:   | |     `-qualTypeDetail: QualType 0x{{.+}} 'const T' const
// CHECK:   | |       `-typeDetails: TemplateTypeParmType 0x{{.+}} 'T' dependent depth 0 index 0
// CHECK:   | |         `-TemplateTypeParm 0x{{.+}} 'T'
// CHECK:   | `-FunctionDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used default_arg_fn2 'int (&(const P2718R0::A &))[3]' implicit_instantiation
// CHECK:   |   |-TemplateArgument type 'P2718R0::A'
// CHECK:   |   | `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   |   |   `-CXXRecord 0x{{.+}} 'A'
// CHECK:   |   `-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} 'const P2718R0::A &' cinit
// CHECK:   |     |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const P2718R0::A' lvalue
// CHECK:   |     | `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const P2718R0::A' lvalue
// CHECK:   |     |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const P2718R0::A' <NoOp>
// CHECK:   |     |     `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |       `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A' 'void ()'
// CHECK:   |     `-typeDetails: LValueReferenceType 0x{{.+}} 'const P2718R0::A &'
// CHECK:   |       `-qualTypeDetail: QualType 0x{{.+}} 'const P2718R0::A' const
// CHECK:   |         `-typeDetails: SubstTemplateTypeParmType 0x{{.+}} 'P2718R0::A' sugar typename depth 0 index 0 T
// CHECK:   |           |-FunctionTemplate 0x{{.+}} 'default_arg_fn2'
// CHECK:   |           `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   |             `-CXXRecord 0x{{.+}} 'A'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test11 'void ()'
// CHECK:   | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   `-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |     |-<<<NULL>>>
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK:   |     |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |     |   | `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |     |   |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int (&(*)(const P2718R0::A &))[3]' <FunctionToPointerDecay>
// CHECK:   |     |   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int (&(const P2718R0::A &))[3]' lvalue Function 0x{{.+}} 'default_arg_fn2' 'int (&(const P2718R0::A &))[3]' (FunctionTemplate 0x{{.+}} 'default_arg_fn2')
// CHECK:   |     |   |   `-CXXDefaultArgExpr 0x{{.+}} <<invalid sloc>> 'const P2718R0::A' lvalue has rewritten init
// CHECK:   |     |   |     `-MaterializeTemporaryExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'const P2718R0::A' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |       `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const P2718R0::A' <NoOp>
// CHECK:   |     |   |         `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |           `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A' 'void ()'
// CHECK:   |     |   `-typeDetails: LValueReferenceType 0x{{.+}} 'int (&)[3]'
// CHECK:   |     |     `-typeDetails: ParenType 0x{{.+}} 'int[3]' sugar
// CHECK:   |     |       `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'int *' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |     |     `-typeDetails: DecayedType 0x{{.+}} 'int *' sugar
// CHECK:   |     |       |-typeDetails: ParenType 0x{{.+}} 'int[3]' sugar
// CHECK:   |     |       | `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |     |       |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |       `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __end1 'int *' cinit
// CHECK:   |     |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int *' '+'
// CHECK:   |     |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'long' 3
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |     |     `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |     |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:   |     | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__end1' 'int *'
// CHECK:   |     |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int *' lvalue prefix '++'
// CHECK:   |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used e 'int' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |     |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK:   |     |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int' sugar
// CHECK:   |     |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     `-CallExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'void'
// CHECK:   |       |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'void (*)(int)' <FunctionToPointerDecay>
// CHECK:   |       | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'void (int)' lvalue Function 0x{{.+}} 'bar' 'void (int)'
// CHECK:   |       `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |         `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'e' 'int'
// CHECK:   |-FunctionTemplateDecl 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> col:{{.*}} foo2
// CHECK:   | |-TemplateTypeParmDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} referenced typename depth 0 index 0 T
// CHECK:   | |-FunctionDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} foo2 'A (const T &, const DefaultA &)'
// CHECK:   | | |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} 'const T &'
// CHECK:   | | | `-typeDetails: LValueReferenceType 0x{{.+}} 'const T &' dependent
// CHECK:   | | |   `-qualTypeDetail: QualType 0x{{.+}} 'const T' const
// CHECK:   | | |     `-typeDetails: TemplateTypeParmType 0x{{.+}} 'T' dependent depth 0 index 0
// CHECK:   | | |       `-TemplateTypeParm 0x{{.+}} 'T'
// CHECK:   | | `-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} Default 'const DefaultA &' cinit
// CHECK:   | |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue
// CHECK:   | |   | `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue
// CHECK:   | |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK:   | |   |     `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' (CXXTemporary 0x{{.+}})
// CHECK:   | |   |       `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK:   | |   `-typeDetails: LValueReferenceType 0x{{.+}} 'const DefaultA &'
// CHECK:   | |     `-qualTypeDetail: QualType 0x{{.+}} 'const DefaultA' const
// CHECK:   | |       `-typeDetails: ElaboratedType 0x{{.+}} 'DefaultA' sugar
// CHECK:   | |         `-typeDetails: RecordType 0x{{.+}} 'P2718R0::DefaultA'
// CHECK:   | |           `-CXXRecord 0x{{.+}} 'DefaultA'
// CHECK:   | `-FunctionDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used foo2 'A (const P2718R0::A &, const DefaultA &)' implicit_instantiation
// CHECK:   |   |-TemplateArgument type 'P2718R0::A'
// CHECK:   |   | `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   |   |   `-CXXRecord 0x{{.+}} 'A'
// CHECK:   |   |-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} 'const P2718R0::A &'
// CHECK:   |   | `-typeDetails: LValueReferenceType 0x{{.+}} 'const P2718R0::A &'
// CHECK:   |   |   `-qualTypeDetail: QualType 0x{{.+}} 'const P2718R0::A' const
// CHECK:   |   |     `-typeDetails: SubstTemplateTypeParmType 0x{{.+}} 'P2718R0::A' sugar typename depth 0 index 0 T
// CHECK:   |   |       |-FunctionTemplate 0x{{.+}} 'foo2'
// CHECK:   |   |       `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:   |   |         `-CXXRecord 0x{{.+}} 'A'
// CHECK:   |   `-ParmVarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} Default 'const DefaultA &' cinit
// CHECK:   |     |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue
// CHECK:   |     | `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue
// CHECK:   |     |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK:   |     |     `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' (CXXTemporary 0x{{.+}})
// CHECK:   |     |       `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK:   |     `-typeDetails: LValueReferenceType 0x{{.+}} 'const DefaultA &'
// CHECK:   |       `-qualTypeDetail: QualType 0x{{.+}} 'const DefaultA' const
// CHECK:   |         `-typeDetails: ElaboratedType 0x{{.+}} 'DefaultA' sugar
// CHECK:   |           `-typeDetails: RecordType 0x{{.+}} 'P2718R0::DefaultA'
// CHECK:   |             `-CXXRecord 0x{{.+}} 'DefaultA'
// CHECK:   |-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test12 'void ()'
// CHECK:   | `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |   `-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:   |     |-<<<NULL>>>
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'int (&)[3]' cinit
// CHECK:   |     |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |     |   | `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int[3]' lvalue
// CHECK:   |     |   |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int (&(*)(const P2718R0::A &))[3]' <FunctionToPointerDecay>
// CHECK:   |     |   |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int (&(const P2718R0::A &))[3]' lvalue Function 0x{{.+}} 'default_arg_fn2' 'int (&(const P2718R0::A &))[3]' (FunctionTemplate 0x{{.+}} 'default_arg_fn2')
// CHECK:   |     |   |   `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |     `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' <NoOp>
// CHECK:   |     |   |       `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |         `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:   |     |   |           |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'A (*)(const P2718R0::A &, const DefaultA &)' <FunctionToPointerDecay>
// CHECK:   |     |   |           | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'A (const P2718R0::A &, const DefaultA &)' lvalue Function 0x{{.+}} 'foo2' 'A (const P2718R0::A &, const DefaultA &)' (FunctionTemplate 0x{{.+}} 'foo2')
// CHECK:   |     |   |           |-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |           | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' <NoOp>
// CHECK:   |     |   |           |   `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |           |     `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:   |     |   |           |       |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'A (*)(const P2718R0::A &, const DefaultA &)' <FunctionToPointerDecay>
// CHECK:   |     |   |           |       | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'A (const P2718R0::A &, const DefaultA &)' lvalue Function 0x{{.+}} 'foo2' 'A (const P2718R0::A &, const DefaultA &)' (FunctionTemplate 0x{{.+}} 'foo2')
// CHECK:   |     |   |           |       |-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |           |       | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' <NoOp>
// CHECK:   |     |   |           |       |   `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |           |       |     `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:   |     |   |           |       |       |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'A (*)(const P2718R0::A &, const DefaultA &)' <FunctionToPointerDecay>
// CHECK:   |     |   |           |       |       | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'A (const P2718R0::A &, const DefaultA &)' lvalue Function 0x{{.+}} 'foo2' 'A (const P2718R0::A &, const DefaultA &)' (FunctionTemplate 0x{{.+}} 'foo2')
// CHECK:   |     |   |           |       |       |-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |           |       |       | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const A':'const P2718R0::A' <NoOp>
// CHECK:   |     |   |           |       |       |   `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |           |       |       |     `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' 'void ()'
// CHECK:   |     |   |           |       |       `-CXXDefaultArgExpr 0x{{.+}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK:   |     |   |           |       |         `-MaterializeTemporaryExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |           |       |           `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK:   |     |   |           |       |             `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |           |       |               `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK:   |     |   |           |       `-CXXDefaultArgExpr 0x{{.+}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK:   |     |   |           |         `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |           |           `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK:   |     |   |           |             `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |           |               `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK:   |     |   |           `-CXXDefaultArgExpr 0x{{.+}} <<invalid sloc>> 'const DefaultA':'const P2718R0::DefaultA' lvalue has rewritten init
// CHECK:   |     |   |             `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' lvalue extended by Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   |               `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'const DefaultA':'const P2718R0::DefaultA' <NoOp>
// CHECK:   |     |   |                 `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' (CXXTemporary 0x{{.+}})
// CHECK:   |     |   |                   `-CXXTemporaryObjectExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'DefaultA':'P2718R0::DefaultA' 'void ()'
// CHECK:   |     |   `-typeDetails: LValueReferenceType 0x{{.+}} 'int (&)[3]'
// CHECK:   |     |     `-typeDetails: ParenType 0x{{.+}} 'int[3]' sugar
// CHECK:   |     |       `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'int *' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |     |     `-typeDetails: DecayedType 0x{{.+}} 'int *' sugar
// CHECK:   |     |       |-typeDetails: ParenType 0x{{.+}} 'int[3]' sugar
// CHECK:   |     |       | `-typeDetails: ConstantArrayType 0x{{.+}} 'int[3]' 3
// CHECK:   |     |       |   `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |       `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |     |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __end1 'int *' cinit
// CHECK:   |     |   |-BinaryOperator 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'int *' '+'
// CHECK:   |     |   | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <ArrayToPointerDecay>
// CHECK:   |     |   | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int[3]' lvalue Var 0x{{.+}} '__range1' 'int (&)[3]'
// CHECK:   |     |   | `-IntegerLiteral 0x{{.+}} <col:{{.*}}> 'long' 3
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int *' sugar
// CHECK:   |     |     `-typeDetails: PointerType 0x{{.+}} 'int *'
// CHECK:   |     |       `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:   |     | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__end1' 'int *'
// CHECK:   |     |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int *' lvalue prefix '++'
// CHECK:   |     | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:   |     | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used e 'int' cinit
// CHECK:   |     |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |     |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'int' lvalue prefix '*' cannot overflow
// CHECK:   |     |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int *' <LValueToRValue>
// CHECK:   |     |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int *' lvalue Var 0x{{.+}} '__begin1' 'int *'
// CHECK:   |     |   `-typeDetails: AutoType 0x{{.+}} 'int' sugar
// CHECK:   |     |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:   |     `-CallExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'void'
// CHECK:   |       |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'void (*)(int)' <FunctionToPointerDecay>
// CHECK:   |       | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'void (int)' lvalue Function 0x{{.+}} 'bar' 'void (int)'
// CHECK:   |       `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:   |         `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'e' 'int'
// CHECK:   `-FunctionDecl 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}> line:{{.*}}:{{.*}} test13 'void ()'
// CHECK:     `-CompoundStmt 0x{{.+}} <col:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:       `-CXXForRangeStmt 0x{{.+}} <line:{{.*}}:{{.*}}, line:{{.*}}:{{.*}}>
// CHECK:         |-<<<NULL>>>
// CHECK:         |-DeclStmt 0x{{.+}} <line:{{.*}}:{{.*}}>
// CHECK:         | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} implicit used __range1 'A &&' cinit
// CHECK:         |   |-ExprWithCleanups 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' xvalue
// CHECK:         |   | `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' xvalue extended by Var 0x{{.+}} '__range1' 'A &&'
// CHECK:         |   |   `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:         |   |     `-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:         |   |       `-MemberExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<bound member function type>' .g 0x{{.+}}
// CHECK:         |   |         `-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' lvalue
// CHECK:         |   |           `-MemberExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<bound member function type>' .r 0x{{.+}}
// CHECK:         |   |             `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' xvalue extended by Var 0x{{.+}} '__range1' 'A &&'
// CHECK:         |   |               `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:         |   |                 `-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:         |   |                   `-MemberExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<bound member function type>' .g 0x{{.+}}
// CHECK:         |   |                     `-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' lvalue
// CHECK:         |   |                       `-MemberExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<bound member function type>' .r 0x{{.+}}
// CHECK:         |   |                         `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' xvalue extended by Var 0x{{.+}} '__range1' 'A &&'
// CHECK:         |   |                           `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:         |   |                             `-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A'
// CHECK:         |   |                               `-MemberExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<bound member function type>' .g 0x{{.+}}
// CHECK:         |   |                                 `-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'A':'P2718R0::A' lvalue
// CHECK:         |   |                                   `-MemberExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> '<bound member function type>' .r 0x{{.+}}
// CHECK:         |   |                                     `-MaterializeTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A' xvalue extended by Var 0x{{.+}} '__range1' 'A &&'
// CHECK:         |   |                                       `-CXXBindTemporaryExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A' (CXXTemporary 0x{{.+}})
// CHECK:         |   |                                         `-CallExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A'
// CHECK:         |   |                                           `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A (*)()' <FunctionToPointerDecay>
// CHECK:         |   |                                             `-DeclRefExpr 0x{{.+}} <col:{{.*}}, col:{{.*}}> 'P2718R0::A ()' lvalue Function 0x{{.+}} 'dg' 'P2718R0::A ()' (FunctionTemplate 0x{{.+}} 'dg')
// CHECK:         |   `-typeDetails: RValueReferenceType 0x{{.+}} 'A &&'
// CHECK:         |     `-typeDetails: AutoType 0x{{.+}} 'A' sugar
// CHECK:         |       `-typeDetails: ElaboratedType 0x{{.+}} 'A' sugar
// CHECK:         |         `-typeDetails: RecordType 0x{{.+}} 'P2718R0::A'
// CHECK:         |           `-CXXRecord 0x{{.+}} 'A'
// CHECK:         |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:         | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __begin1 'const int *' cinit
// CHECK:         |   |-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}> 'const int *'
// CHECK:         |   | `-MemberExpr 0x{{.+}} <col:{{.*}}> '<bound member function type>' .begin 0x{{.+}}
// CHECK:         |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::A' lvalue <NoOp>
// CHECK:         |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'A':'P2718R0::A' lvalue Var 0x{{.+}} '__range1' 'A &&'
// CHECK:         |   `-typeDetails: AutoType 0x{{.+}} 'const int *' sugar
// CHECK:         |     `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:         |       `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |-DeclStmt 0x{{.+}} <col:{{.*}}>
// CHECK:         | `-VarDecl 0x{{.+}} <col:{{.*}}> col:{{.*}} implicit used __end1 'const int *' cinit
// CHECK:         |   |-CXXMemberCallExpr 0x{{.+}} <col:{{.*}}> 'const int *'
// CHECK:         |   | `-MemberExpr 0x{{.+}} <col:{{.*}}> '<bound member function type>' .end 0x{{.+}}
// CHECK:         |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const P2718R0::A' lvalue <NoOp>
// CHECK:         |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'A':'P2718R0::A' lvalue Var 0x{{.+}} '__range1' 'A &&'
// CHECK:         |   `-typeDetails: AutoType 0x{{.+}} 'const int *' sugar
// CHECK:         |     `-typeDetails: PointerType 0x{{.+}} 'const int *'
// CHECK:         |       `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
// CHECK:         |         `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         |-BinaryOperator 0x{{.+}} <col:{{.*}}> 'bool' '!='
// CHECK:         | |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <LValueToRValue>
// CHECK:         | | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__begin1' 'const int *'
// CHECK:         | `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <LValueToRValue>
// CHECK:         |   `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__end1' 'const int *'
// CHECK:         |-UnaryOperator 0x{{.+}} <col:{{.*}}> 'const int *' lvalue prefix '++'
// CHECK:         | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__begin1' 'const int *'
// CHECK:         |-DeclStmt 0x{{.+}} <col:{{.*}}, col:{{.*}}>
// CHECK:         | `-VarDecl 0x{{.+}} <col:{{.*}}, col:{{.*}}> col:{{.*}} used e 'int' cinit
// CHECK:         |   |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:         |   | `-UnaryOperator 0x{{.+}} <col:{{.*}}> 'const int' lvalue prefix '*' cannot overflow
// CHECK:         |   |   `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'const int *' <LValueToRValue>
// CHECK:         |   |     `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'const int *' lvalue Var 0x{{.+}} '__begin1' 'const int *'
// CHECK:         |   `-typeDetails: AutoType 0x{{.+}} 'int' sugar
// CHECK:         |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
// CHECK:         `-CallExpr 0x{{.+}} <line:{{.*}}:{{.*}}, col:{{.*}}> 'void'
// CHECK:           |-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'void (*)(int)' <FunctionToPointerDecay>
// CHECK:           | `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'void (int)' lvalue Function 0x{{.+}} 'bar' 'void (int)'
// CHECK:           `-ImplicitCastExpr 0x{{.+}} <col:{{.*}}> 'int' <LValueToRValue>
// CHECK:             `-DeclRefExpr 0x{{.+}} <col:{{.*}}> 'int' lvalue Var 0x{{.+}} 'e' 'int'