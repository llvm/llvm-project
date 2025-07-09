// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -fcxx-exceptions -std=gnu++17 -frecovery-ast -frecovery-ast-type -ast-dump %s | FileCheck %s

int some_func(int *);

// CHECK: |-FunctionDecl {{.*}} some_func 'int (int *)'
// CHECK: | `-ParmVarDecl {{.*}} 'int *'
// CHECK: | `-typeDetails: PointerType {{.*}} 'int *'
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

int invalid_call = some_func(123);

// CHECK: |-VarDecl {{.*}} invalid_call 'int' cinit
// CHECK: | |-RecoveryExpr {{.*}} 'int' contains-errors
// CHECK: | | |-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'some_func' {{.*}}
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 123
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

void test_invalid_call(int s) {
 some_func(undef1, undef2+1);
 s = some_func(undef1);
 int var = some_func(undef1);
}

// CHECK: |-FunctionDecl {{.*}} used test_invalid_call 'void (int)'
// CHECK: | |-ParmVarDecl {{.*}} used s 'int'
// CHECK: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK: | `-CompoundStmt {{.*}} 
// CHECK: | |-CallExpr {{.*}} '<dependent type>' contains-errors
// CHECK: | | |-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'some_func' {{.*}}
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-BinaryOperator {{.*}} '<dependent type>' contains-errors '+'
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 1
// CHECK: | |-BinaryOperator {{.*}} '<dependent type>' contains-errors '='
// CHECK: | | |-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 's' 'int'
// CHECK: | | `-CallExpr {{.*}} '<dependent type>' contains-errors
// CHECK: | | |-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'some_func' {{.*}}
// CHECK: | | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | `-DeclStmt {{.*}} 
// CHECK: | `-VarDecl {{.*}} var 'int' cinit
// CHECK: | |-CallExpr {{.*}} '<dependent type>' contains-errors
// CHECK: | | |-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'some_func' {{.*}}
// CHECK: | | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

int ambig_func(double);

// CHECK: |-FunctionDecl {{.*}}
// CHECK: | `-ParmVarDecl {{.*}} 
// CHECK: | `-typeDetails: BuiltinType {{.*}}

int ambig_func(float);

// CHECK: |-FunctionDecl {{.*}} 
// CHECK: | `-ParmVarDecl {{.*}}
// CHECK: | `-typeDetails: BuiltinType {{.*}}

int ambig_call = ambig_func(123);

// CHECK: |-VarDecl {{.*}} ambig_call 'int' cinit
// CHECK: | |-RecoveryExpr {{.*}} 'int' contains-errors
// CHECK: | | |-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'ambig_func' {{.*}} {{.*}}
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 123
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

int unresolved_call1 = bar();

// CHECK: |-VarDecl {{.*}} unresolved_call1 'int' cinit
// CHECK: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'bar' empty
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

int unresolved_call2 = bar(baz(), qux());

// CHECK: |-VarDecl {{.*}} unresolved_call2 'int' cinit
// CHECK: | |-CallExpr {{.*}} '<dependent type>' contains-errors
// CHECK: | | |-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'bar' empty
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'baz' empty
// CHECK: | | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'qux' empty
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

constexpr int a = 10;

// CHECK: |-VarDecl {{.*}} used a 'const int' constexpr cinit
// CHECK: | |-value: Int 10
// CHECK: | |-IntegerLiteral {{.*}} 'int' 10
// CHECK: | `-qualTypeDetail: QualType {{.*}} 'const int' const
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

int postfix_inc = a++;

// CHECK: |-VarDecl {{.*}} postfix_inc 'int' cinit
// CHECK: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-DeclRefExpr {{.*}} 'const int' lvalue Var {{.*}} 'a' 'const int'
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

int prefix_inc = ++a;

// CHECK: |-VarDecl {{.*}} prefix_inc 'int' cinit
// CHECK: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-DeclRefExpr {{.*}} 'const int' lvalue Var {{.*}} 'a' 'const int'
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

int unary_address = &(a + 1);
// CHECK: |-VarDecl {{.*}} unary_address 'int' cinit
// CHECK: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-ParenExpr {{.*}} 'int'
// CHECK: | | `-BinaryOperator {{.*}} 'int' '+'
// CHECK: | | |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK: | | | `-DeclRefExpr {{.*}} 'const int' lvalue Var {{.*}} 'a' 'const int' non_odr_use_constant
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 1
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

int unary_bitinverse = ~(a + 0.0);
// CHECK: |-VarDecl {{.*}} unary_bitinverse 'int' cinit
// CHECK: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-ParenExpr {{.*}} 'double'
// CHECK: | | `-BinaryOperator {{.*}} 'double' '+'
// CHECK: | | |-ImplicitCastExpr {{.*}} 'double' <IntegralToFloating>
// CHECK: | | | `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK: | | | `-DeclRefExpr {{.*}} 'const int' lvalue Var {{.*}} 'a' 'const int' non_odr_use_constant
// CHECK: | | `-FloatingLiteral {{.*}} 'double' 0.000000e+00
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

int binary = a + nullptr;
// CHECK: |-VarDecl {{.*}} binary 'int' cinit
// CHECK: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | |-DeclRefExpr {{.*}} 'const int' lvalue Var {{.*}} 'a' 'const int'
// CHECK: | | `-CXXNullPtrLiteralExpr {{.*}} 'std::nullptr_t'
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

int ternary = a ? nullptr : a;

// CHECK: |-VarDecl {{.*}} ternary 'int' cinit
// CHECK: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | |-DeclRefExpr {{.*}} 'const int' lvalue Var {{.*}} 'a' 'const int'
// CHECK: | | |-CXXNullPtrLiteralExpr {{.*}} 'std::nullptr_t'
// CHECK: | | `-DeclRefExpr {{.*}} 'const int' lvalue Var {{.*}} 'a' 'const int'
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

struct Foo {} foo;

// CHECK: |-CXXRecordDecl {{.*}} struct Foo definition
// CHECK: | |-DefinitionData pass_in_registers empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK: | | |-DefaultConstructor exists trivial constexpr defaulted_is_constexpr
// CHECK: | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK: | | |-MoveConstructor exists simple trivial
// CHECK: | | |-CopyAssignment simple trivial has_const_param implicit_has_const_param
// CHECK: | | |-MoveAssignment exists simple trivial
// CHECK: | | `-Destructor simple irrelevant trivial
// CHECK: | |-CXXRecordDecl {{.*}} implicit struct Foo
// CHECK: | |-CXXConstructorDecl {{.*}} implicit used constexpr Foo 'void () noexcept' inline default trivial
// CHECK: | | `-CompoundStmt {{.*}} 
// CHECK: | |-CXXConstructorDecl {{.*}} implicit constexpr Foo 'void (const Foo &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK: | | `-ParmVarDecl {{.*}} 'const Foo &'
// CHECK: | | `-typeDetails: LValueReferenceType {{.*}} 'const Foo &'
// CHECK: | | `-qualTypeDetail: QualType {{.*}} 'const Foo' const
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Foo' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Foo'
// CHECK: | | `-CXXRecord {{.*}} 'Foo'
// CHECK: | |-CXXConstructorDecl {{.*}} implicit constexpr Foo 'void (Foo &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK: | | `-ParmVarDecl {{.*}} 'Foo &&'
// CHECK: | | `-typeDetails: RValueReferenceType {{.*}} 'Foo &&'
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Foo' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Foo'
// CHECK: | | `-CXXRecord {{.*}} 'Foo'
// CHECK: | |-CXXMethodDecl {{.*}} implicit constexpr operator= 'Foo &(const Foo &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK: | | `-ParmVarDecl {{.*}} 'const Foo &'
// CHECK: | | `-typeDetails: LValueReferenceType {{.*}} 'const Foo &'
// CHECK: | | `-qualTypeDetail: QualType {{.*}} 'const Foo' const
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Foo' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Foo'
// CHECK: | | `-CXXRecord {{.*}} 'Foo'
// CHECK: | |-CXXMethodDecl {{.*}} implicit constexpr operator= 'Foo &(Foo &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK: | | `-ParmVarDecl {{.*}} 'Foo &&'
// CHECK: | | `-typeDetails: RValueReferenceType {{.*}} 'Foo &&'
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Foo' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Foo'
// CHECK: | | `-CXXRecord {{.*}} 'Foo'
// CHECK: | `-CXXDestructorDecl {{.*}} implicit ~Foo 'void ()' inline default trivial noexcept-unevaluated {{.*}}
// CHECK: |-VarDecl {{.*}} used foo 'struct Foo':'Foo' callinit
// CHECK: | |-CXXConstructExpr {{.*}} 'struct Foo':'Foo' 'void () noexcept'
// CHECK: | `-typeDetails: ElaboratedType {{.*}} 'struct Foo' sugar
// CHECK: | `-typeDetails: RecordType {{.*}} 'Foo'
// CHECK: | `-CXXRecord {{.*}} 'Foo'

void test(int x) {
 foo.abc;
 foo->func(x);
}


// CHECK: |-FunctionDecl {{.*}} test 'void (int)'
// CHECK: | |-ParmVarDecl {{.*}} used x 'int'
// CHECK: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK: | `-CompoundStmt {{.*}} 
// CHECK: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-DeclRefExpr {{.*}} 'struct Foo':'Foo' lvalue Var {{.*}} 'foo' 'struct Foo':'Foo'
// CHECK: | `-CallExpr {{.*}} '<dependent type>' contains-errors
// CHECK: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-DeclRefExpr {{.*}} 'struct Foo':'Foo' lvalue Var {{.*}} 'foo' 'struct Foo':'Foo'
// CHECK: | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int'

void AccessIncompleteClass() {
 struct Forward;
 Forward* ptr;
 ptr->method();
}

// CHECK: |-FunctionDecl {{.*}} AccessIncompleteClass 'void ()'
// CHECK: | `-CompoundStmt {{.*}} 
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-CXXRecordDecl {{.*}} referenced struct Forward
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} used ptr 'Forward *'
// CHECK: | | `-typeDetails: PointerType {{.*}} 'Forward *'
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Forward' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Forward'
// CHECK: | | `-CXXRecord {{.*}} 'Forward'
// CHECK: | `-CallExpr {{.*}} '<dependent type>' contains-errors
// CHECK: | `-CXXDependentScopeMemberExpr {{.*}} '<dependent type>' contains-errors lvalue ->method
// CHECK: | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | `-DeclRefExpr {{.*}} 'Forward *' lvalue Var {{.*}} 'ptr' 'Forward *'

struct Foo2 {
 double func();
 class ForwardClass;
 ForwardClass createFwd();

 int overload();
 int overload(int, int);
};

// CHECK: |-CXXRecordDecl {{.*}} referenced struct Foo2 definition
// CHECK: | |-DefinitionData pass_in_registers empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK: | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// CHECK: | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK: | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK: | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK: | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK: | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK: | |-CXXRecordDecl {{.*}} implicit struct Foo2
// CHECK: | |-CXXMethodDecl {{.*}} used func 'double ()'
// CHECK: | |-CXXRecordDecl {{.*}} referenced class ForwardClass
// CHECK: | |-CXXMethodDecl {{.*}} used createFwd 'ForwardClass ()'
// CHECK: | |-CXXMethodDecl {{.*}} overload 'int ()'
// CHECK: | `-CXXMethodDecl {{.*}} overload 'int (int, int)'
// CHECK: | |-ParmVarDecl {{.*}} 'int'
// CHECK: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK: | `-ParmVarDecl {{.*}} 'int'
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

void test2(Foo2 f) {
 f.func(1);
 f.createFwd();
 f.overload(1);
}

// CHECK: |-FunctionDecl {{.*}} test2 'void (Foo2)'
// CHECK: | |-ParmVarDecl {{.*}} used f 'Foo2'
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Foo2' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Foo2'
// CHECK: | | `-CXXRecord {{.*}} 'Foo2'
// CHECK: | `-CompoundStmt {{.*}} 
// CHECK: | |-RecoveryExpr {{.*}} 'double' contains-errors
// CHECK: | | |-MemberExpr {{.*}} '<bound member function type>' .func {{.*}}
// CHECK: | | | `-DeclRefExpr {{.*}} 'Foo2' lvalue ParmVar {{.*}} 'f' 'Foo2'
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 1
// CHECK: | |-RecoveryExpr {{.*}} 'ForwardClass':'Foo2::ForwardClass' contains-errors
// CHECK: | | `-MemberExpr {{.*}} '<bound member function type>' .createFwd {{.*}}
// CHECK: | | `-DeclRefExpr {{.*}} 'Foo2' lvalue ParmVar {{.*}} 'f' 'Foo2'
// CHECK: | `-RecoveryExpr {{.*}} 'int' contains-errors
// CHECK: | |-UnresolvedMemberExpr {{.*}} '<bound member function type>' lvalue
// CHECK: | | `-DeclRefExpr {{.*}} 'Foo2' lvalue ParmVar {{.*}} 'f' 'Foo2'
// CHECK: | `-IntegerLiteral {{.*}} 'int' 1

struct alignas(invalid()) Aligned {};

// CHECK: |-CXXRecordDecl {{.*}} struct Aligned definition
// CHECK: | |-DefinitionData pass_in_registers empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK: | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// CHECK: | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK: | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK: | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK: | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK: | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK: | |-attrDetails: AlignedAttr {{.*}} alignas
// CHECK: | | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'invalid' empty
// CHECK: | `-CXXRecordDecl {{.*}} implicit struct Aligned

auto f();

// CHECK: |-FunctionDecl {{.*}} f 'auto ()'

int f(double);

// CHECK: |-FunctionDecl {{.*}} f 'int (double)'
// CHECK: | `-ParmVarDecl {{.*}} 'double'
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'double'

int unknown_type_call = f(0, 0);

// CHECK: |-VarDecl {{.*}} unknown_type_call 'int' cinit
// CHECK: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | |-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'f' {{.*}} {{.*}}
// CHECK: | | |-IntegerLiteral {{.*}} 'int' 0
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 0
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

void InvalidInitalizer(int x) {
 struct Bar { Bar(); };
 Bar a1(1);
 Bar a2(x);
 Bar a3{x};
 Bar a4(invalid());
 Bar a5{invalid()};

 Bar b1 = 1;
 Bar b2 = {1};
 Bar b3 = Bar(x);
 Bar b4 = Bar{x};
 Bar b5 = Bar(invalid());
 Bar b6 = Bar{invalid()};

 Bar(1);

 int var1 = undef + 1;
}

// CHECK: |-FunctionDecl {{.*}} InvalidInitalizer 'void (int)'
// CHECK: | |-ParmVarDecl {{.*}} used x 'int'
// CHECK: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK: | `-CompoundStmt {{.*}} 
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-CXXRecordDecl {{.*}} referenced struct Bar definition
// CHECK: | | |-DefinitionData pass_in_registers empty standard_layout trivially_copyable has_user_declared_ctor can_const_default_init
// CHECK: | | | |-DefaultConstructor exists non_trivial user_provided defaulted_is_constexpr
// CHECK: | | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK: | | | |-MoveConstructor exists simple trivial
// CHECK: | | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK: | | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK: | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK: | | |-CXXRecordDecl {{.*}} implicit referenced struct Bar
// CHECK: | | |-CXXConstructorDecl {{.*}} Bar 'void ()'
// CHECK: | | |-CXXConstructorDecl {{.*}} implicit constexpr Bar 'void (const Bar &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK: | | | `-ParmVarDecl {{.*}} 'const Bar &'
// CHECK: | | | `-typeDetails: LValueReferenceType {{.*}} 'const Bar &'
// CHECK: | | | `-qualTypeDetail: QualType {{.*}} 'const Bar' const
// CHECK: | | | `-typeDetails: ElaboratedType {{.*}} 'Bar' sugar
// CHECK: | | | `-typeDetails: RecordType {{.*}} 'Bar'
// CHECK: | | | `-CXXRecord {{.*}} 'Bar'
// CHECK: | | `-CXXConstructorDecl {{.*}} implicit constexpr Bar 'void (Bar &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK: | | `-ParmVarDecl {{.*}} 'Bar &&'
// CHECK: | | `-typeDetails: RValueReferenceType {{.*}} 'Bar &&'
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Bar' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Bar'
// CHECK: | | `-CXXRecord {{.*}} 'Bar'
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} a1 'Bar' cinit
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-IntegerLiteral {{.*}} 'int' 1
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Bar' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Bar'
// CHECK: | | `-CXXRecord {{.*}} 'Bar'
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} a2 'Bar' cinit
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Bar' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Bar'
// CHECK: | | `-CXXRecord {{.*}} 'Bar'
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} a3 'Bar' cinit
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-InitListExpr {{.*}} 'void'
// CHECK: | | | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Bar' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Bar'
// CHECK: | | `-CXXRecord {{.*}} 'Bar'
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} a4 'Bar' callinit
// CHECK: | | |-ParenListExpr {{.*}} 'NULL TYPE' contains-errors
// CHECK: | | | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'invalid' empty
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Bar' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Bar'
// CHECK: | | `-CXXRecord {{.*}} 'Bar'
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} a5 'Bar' listinit
// CHECK: | | |-InitListExpr {{.*}} 'void' contains-errors
// CHECK: | | | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'invalid' empty
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Bar' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Bar'
// CHECK: | | `-CXXRecord {{.*}} 'Bar'
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} b1 'Bar' cinit
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-IntegerLiteral {{.*}} 'int' 1
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Bar' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Bar'
// CHECK: | | `-CXXRecord {{.*}} 'Bar'
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} b2 'Bar' cinit
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-InitListExpr {{.*}} 'void'
// CHECK: | | | `-IntegerLiteral {{.*}} 'int' 1
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Bar' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Bar'
// CHECK: | | `-CXXRecord {{.*}} 'Bar'
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} b3 'Bar' cinit
// CHECK: | | |-RecoveryExpr {{.*}} 'Bar' contains-errors
// CHECK: | | | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Bar' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Bar'
// CHECK: | | `-CXXRecord {{.*}} 'Bar'
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} b4 'Bar' cinit
// CHECK: | | |-RecoveryExpr {{.*}} 'Bar' contains-errors
// CHECK: | | | `-InitListExpr {{.*}} 'void'
// CHECK: | | | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'x' 'int'
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Bar' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Bar'
// CHECK: | | `-CXXRecord {{.*}} 'Bar'
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} b5 'Bar' cinit
// CHECK: | | |-CXXUnresolvedConstructExpr {{.*}} 'Bar' contains-errors 'Bar'
// CHECK: | | | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'invalid' empty
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Bar' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Bar'
// CHECK: | | `-CXXRecord {{.*}} 'Bar'
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} b6 'Bar' cinit
// CHECK: | | |-CXXUnresolvedConstructExpr {{.*}} 'Bar' contains-errors 'Bar' list
// CHECK: | | | `-InitListExpr {{.*}} 'void' contains-errors
// CHECK: | | | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'invalid' empty
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'Bar' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'Bar'
// CHECK: | | `-CXXRecord {{.*}} 'Bar'
// CHECK: | |-RecoveryExpr {{.*}} 'Bar' contains-errors
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 1
// CHECK: | `-DeclStmt {{.*}} 
// CHECK: | `-VarDecl {{.*}} var1 'int' cinit
// CHECK: | |-BinaryOperator {{.*}} '<dependent type>' contains-errors '+'
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 1
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

void InitializerForAuto() {
 auto a = invalid();
 auto b = some_func(invalid());

 decltype(ned);
 auto unresolved_typo = gned.*[] {};
}

// CHECK: |-FunctionDecl {{.*}} InitializerForAuto 'void ()'
// CHECK: | `-CompoundStmt {{.*}} 
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} invalid a 'auto' cinit
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'invalid' empty
// CHECK: | | `-typeDetails: AutoType {{.*}} 'auto' undeduced
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} invalid b 'auto' cinit
// CHECK: | | |-CallExpr {{.*}} '<dependent type>' contains-errors
// CHECK: | | | |-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'some_func' {{.*}}
// CHECK: | | | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'invalid' empty
// CHECK: | | `-typeDetails: AutoType {{.*}} 'auto' undeduced
// CHECK: | `-DeclStmt {{.*}} 
// CHECK: | `-VarDecl {{.*}} invalid unresolved_typo 'auto'
// CHECK: | `-typeDetails: AutoType {{.*}} 'auto' undeduced


using Escape = decltype([] { return undef(); }());

// CHECK: |-CXXRecordDecl {{.*}} implicit class definition
// CHECK: | |-DefinitionData lambda pass_in_registers empty standard_layout trivially_copyable literal can_const_default_init
// CHECK: | | |-DefaultConstructor defaulted_is_constexpr
// CHECK: | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK: | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK: | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK: | | |-MoveAssignment
// CHECK: | | `-Destructor simple irrelevant trivial
// CHECK: | |-CXXMethodDecl {{.*}} invalid operator() 'auto () const -> auto' inline
// CHECK: | | `-CompoundStmt {{.*}} 
// CHECK: | |-CXXConversionDecl {{.*}} implicit constexpr operator auto (*)() 'auto (*() const noexcept)() -> auto' inline
// CHECK: | |-CXXMethodDecl {{.*}} implicit __invoke 'auto () -> auto' static inline
// CHECK: | `-CXXDestructorDecl {{.*}} 


struct {
 int& abc;
} NoCrashOnInvalidInitList = {
 .abc = nullptr,
};

// CHECK: |-CXXRecordDecl {{.*}} struct definition
// CHECK: | |-DefinitionData pass_in_registers aggregate trivially_copyable trivial literal
// CHECK: | | |-DefaultConstructor exists trivial needs_implicit
// CHECK: | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK: | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK: | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK: | | |-MoveAssignment exists trivial needs_implicit
// CHECK: | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK: | `-FieldDecl {{.*}} abc 'int &'
// CHECK: |-VarDecl {{.*}} NoCrashOnInvalidInitList 
// CHECK: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-InitListExpr {{.*}} 'struct (unnamed struct
// CHECK: | | `-DesignatedInitExpr {{.*}} 'void'
// CHECK: | | `-CXXNullPtrLiteralExpr {{.*}} 'std::nullptr_t'
// CHECK: | `-typeDetails: ElaboratedType {{.*}} 'struct
// CHECK: | `-typeDetails: RecordType {{.*}}
// CHECK: | `-CXXRecord {{.*}} ''


// Verify the value category of recovery expression.
int prvalue(int);

// CHECK: |-FunctionDecl {{.*}} prvalue 'int (int)'
// CHECK: | `-ParmVarDecl {{.*}} 'int'
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

int &lvalue(int);

// CHECK: |-FunctionDecl {{.*}} lvalue 'int &(int)'
// CHECK: | `-ParmVarDecl {{.*}} 'int'
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

int &&xvalue(int);

// CHECK: |-FunctionDecl {{.*}} xvalue 'int &&(int)'
// CHECK: | `-ParmVarDecl {{.*}} 'int'
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

void ValueCategory() {
 prvalue(); // call to a function (nonreference return type) yields a prvalue (not print by default)
 lvalue(); // call to a function (lvalue reference return type) yields an lvalue.
 xvalue(); // call to a function (rvalue reference return type) yields an xvalue.
}

// CHECK: |-FunctionDecl {{.*}} ValueCategory 'void ()'
// CHECK: | `-CompoundStmt {{.*}} 
// CHECK: | |-RecoveryExpr {{.*}} 'int' contains-errors
// CHECK: | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'prvalue' {{.*}}
// CHECK: | |-RecoveryExpr {{.*}} 'int' contains-errors lvalue
// CHECK: | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'lvalue' {{.*}}
// CHECK: | `-RecoveryExpr {{.*}} 'int' contains-errors xvalue
// CHECK: | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'xvalue' {{.*}}

void InvalidCondition() {
 if (invalid()) {}

 while (invalid()) {}

 switch(invalid()) {
 case 1:
 break;
 }

 invalid() ? 1 : 2;
}


// CHECK: |-FunctionDecl {{.*}} InvalidCondition 'void ()'
// CHECK: | `-CompoundStmt {{.*}} 
// CHECK: | |-IfStmt {{.*}} 
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'invalid' empty
// CHECK: | | `-CompoundStmt {{.*}} 
// CHECK: | |-WhileStmt {{.*}} 
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'invalid' empty
// CHECK: | | `-CompoundStmt {{.*}} 
// CHECK: | |-SwitchStmt {{.*}} 
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'invalid' empty
// CHECK: | | `-CompoundStmt {{.*}} 
// CHECK: | | `-CaseStmt {{.*}} 
// CHECK: | | |-IntegerLiteral {{.*}} 'int' 1
// CHECK: | | `-BreakStmt {{.*}} 
// CHECK: | `-ConditionalOperator {{.*}} '<dependent type>' contains-errors
// CHECK: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'invalid' empty
// CHECK: | |-IntegerLiteral {{.*}} 'int' 1
// CHECK: | `-IntegerLiteral {{.*}} 'int' 2

void CtorInitializer() {
 struct S{int m};
 class MemberInit {
 int x, y, z;
 S s;
 MemberInit() : x(invalid), y(invalid, invalid), z(invalid()), s(1,2) {}
 };

 class BaseInit : S {
 BaseInit(float) : S("no match") {}

 BaseInit(double) : S(invalid) {}
 };

 class DelegatingInit {
 DelegatingInit(float) : DelegatingInit("no match") {}
 DelegatingInit(double) : DelegatingInit(invalid) {}
 };
}

// CHECK: |-FunctionDecl {{.*}} CtorInitializer 'void ()'
// CHECK: | `-CompoundStmt {{.*}} 
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-CXXRecordDecl {{.*}} referenced struct S definition
// CHECK: | | |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK: | | | |-DefaultConstructor exists trivial
// CHECK: | | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK: | | | |-MoveConstructor exists simple trivial
// CHECK: | | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK: | | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK: | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK: | | |-CXXRecordDecl {{.*}} implicit referenced struct S
// CHECK: | | |-FieldDecl {{.*}} m 'int'
// CHECK: | | |-CXXConstructorDecl {{.*}} implicit S 'void ()' inline default trivial noexcept-unevaluated {{.*}}
// CHECK: | | |-CXXConstructorDecl {{.*}} implicit constexpr S 'void (const S &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK: | | | `-ParmVarDecl {{.*}} 'const S &'
// CHECK: | | | `-typeDetails: LValueReferenceType {{.*}} 'const S &'
// CHECK: | | | `-qualTypeDetail: QualType {{.*}} 'const S' const
// CHECK: | | | `-typeDetails: ElaboratedType {{.*}} 'S' sugar
// CHECK: | | | `-typeDetails: RecordType {{.*}} 'S'
// CHECK: | | | `-CXXRecord {{.*}} 'S'
// CHECK: | | `-CXXConstructorDecl {{.*}} implicit constexpr S 'void (S &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK: | | `-ParmVarDecl {{.*}} 'S &&'
// CHECK: | | `-typeDetails: RValueReferenceType {{.*}} 'S &&'
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'S' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'S'
// CHECK: | | `-CXXRecord {{.*}} 'S'
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-CXXRecordDecl {{.*}} class MemberInit definition
// CHECK: | | |-DefinitionData pass_in_registers standard_layout trivially_copyable has_user_declared_ctor can_const_default_init
// CHECK: | | | |-DefaultConstructor exists non_trivial user_provided
// CHECK: | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK: | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK: | | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK: | | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK: | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK: | | |-CXXRecordDecl {{.*}} implicit referenced class MemberInit
// CHECK: | | |-FieldDecl {{.*}} x 'int'
// CHECK: | | |-FieldDecl {{.*}} y 'int'
// CHECK: | | |-FieldDecl {{.*}} z 'int'
// CHECK: | | |-FieldDecl {{.*}} s 'S'
// CHECK: | | `-CXXConstructorDecl {{.*}} MemberInit 'void ()' implicit-inline
// CHECK: | | |-CXXCtorInitializer Field {{.*}} 'x' 'int'
// CHECK: | | | `-ParenListExpr {{.*}} 'NULL TYPE' contains-errors
// CHECK: | | | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | |-CXXCtorInitializer Field {{.*}} 'y' 'int'
// CHECK: | | | `-ParenListExpr {{.*}} 'NULL TYPE' contains-errors
// CHECK: | | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | |-CXXCtorInitializer Field {{.*}} 'z' 'int'
// CHECK: | | | `-ParenListExpr {{.*}} 'NULL TYPE' contains-errors
// CHECK: | | | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'invalid' empty
// CHECK: | | |-CXXCtorInitializer Field {{.*}} 's' 'S'
// CHECK: | | | `-RecoveryExpr {{.*}} 'S' contains-errors
// CHECK: | | | |-IntegerLiteral {{.*}} 'int' 1
// CHECK: | | | `-IntegerLiteral {{.*}} 'int' 2
// CHECK: | | `-CompoundStmt {{.*}} 
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-CXXRecordDecl {{.*}} class BaseInit definition
// CHECK: | | |-DefinitionData pass_in_registers standard_layout trivially_copyable has_user_declared_ctor can_const_default_init
// CHECK: | | | |-DefaultConstructor
// CHECK: | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK: | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK: | | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK: | | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK: | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK: | | |-private 'S'
// CHECK: | | |-CXXRecordDecl {{.*}} implicit referenced class BaseInit
// CHECK: | | |-CXXConstructorDecl {{.*}} BaseInit 'void (float)' implicit-inline
// CHECK: | | | |-ParmVarDecl {{.*}} 'float'
// CHECK: | | | | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK: | | | |-CXXCtorInitializer 'S'
// CHECK: | | | | `-RecoveryExpr {{.*}} 'S' contains-errors
// CHECK: | | | | `-StringLiteral {{.*}} 'const char[9]' lvalue "no match"
// CHECK: | | | `-CompoundStmt {{.*}} 
// CHECK: | | `-CXXConstructorDecl {{.*}} BaseInit 'void (double)' implicit-inline
// CHECK: | | |-ParmVarDecl {{.*}} 'double'
// CHECK: | | | `-typeDetails: BuiltinType {{.*}} 'double'
// CHECK: | | |-CXXCtorInitializer 'S'
// CHECK: | | | `-ParenListExpr {{.*}} 'NULL TYPE' contains-errors
// CHECK: | | | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-CompoundStmt {{.*}} 
// CHECK: | `-DeclStmt {{.*}} 
// CHECK: | `-CXXRecordDecl {{.*}} class DelegatingInit definition
// CHECK: | |-DefinitionData pass_in_registers empty standard_layout trivially_copyable has_user_declared_ctor can_const_default_init
// CHECK: | | |-DefaultConstructor defaulted_is_constexpr
// CHECK: | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK: | | |-MoveConstructor exists simple trivial
// CHECK: | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK: | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK: | | `-Destructor simple irrelevant trivial
// CHECK: | |-CXXRecordDecl {{.*}} implicit referenced class DelegatingInit
// CHECK: | |-CXXConstructorDecl {{.*}} DelegatingInit 'void (float)' implicit-inline
// CHECK: | | |-ParmVarDecl {{.*}} 'float'
// CHECK: | | | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK: | | |-CXXCtorInitializer 'DelegatingInit'
// CHECK: | | | `-RecoveryExpr {{.*}} 'DelegatingInit' contains-errors
// CHECK: | | | `-StringLiteral {{.*}} 'const char[9]' lvalue "no match"
// CHECK: | | `-CompoundStmt {{.*}} 
// CHECK: | |-CXXConstructorDecl {{.*}} DelegatingInit 'void (double)' implicit-inline
// CHECK: | | |-ParmVarDecl {{.*}} 'double'
// CHECK: | | | `-typeDetails: BuiltinType {{.*}} 'double'
// CHECK: | | |-CXXCtorInitializer 'DelegatingInit'
// CHECK: | | | `-ParenListExpr {{.*}} 'NULL TYPE' contains-errors
// CHECK: | | | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-CompoundStmt {{.*}} 
// CHECK: | |-CXXConstructorDecl {{.*}} implicit constexpr DelegatingInit 'void (const DelegatingInit &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK: | | `-ParmVarDecl {{.*}} 'const DelegatingInit &'
// CHECK: | | `-typeDetails: LValueReferenceType {{.*}} 'const DelegatingInit &'
// CHECK: | | `-qualTypeDetail: QualType {{.*}} 'const DelegatingInit' const
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'DelegatingInit' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'DelegatingInit'
// CHECK: | | `-CXXRecord {{.*}} 'DelegatingInit'
// CHECK: | |-CXXConstructorDecl {{.*}} implicit constexpr DelegatingInit 'void (DelegatingInit &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK: | | `-ParmVarDecl {{.*}} 'DelegatingInit &&'
// CHECK: | | `-typeDetails: RValueReferenceType {{.*}} 'DelegatingInit &&'
// CHECK: | | `-typeDetails: ElaboratedType {{.*}} 'DelegatingInit' sugar
// CHECK: | | `-typeDetails: RecordType {{.*}} 'DelegatingInit'
// CHECK: | | `-CXXRecord {{.*}} 'DelegatingInit'
// CHECK: | `-CXXDestructorDecl {{.*}} implicit referenced ~DelegatingInit 'void () noexcept' inline default trivial


float *brokenReturn() {
 return 42;
}

// CHECK: |-FunctionDecl {{.*}} brokenReturn 'float *()'
// CHECK: | `-CompoundStmt {{.*}} 
// CHECK: | `-ReturnStmt {{.*}} 
// CHECK: | `-RecoveryExpr {{.*}} 'float *' contains-errors
// CHECK: | `-IntegerLiteral {{.*}} 'int' 42

// Return deduction treats the first, second *and* third differently!
auto *brokenDeducedReturn(int *x, float *y, double *z) {
 if (x) return x;
 if (y) return y;
 if (z) return z;
 return x;
}

// CHECK: |-FunctionDecl {{.*}} invalid brokenDeducedReturn 'int *(int *, float *, double *)'
// CHECK: | |-ParmVarDecl {{.*}} used x 'int *'
// CHECK: | | `-typeDetails: PointerType {{.*}} 'int *'
// CHECK: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK: | |-ParmVarDecl {{.*}} used y 'float *'
// CHECK: | | `-typeDetails: PointerType {{.*}} 'float *'
// CHECK: | | `-typeDetails: BuiltinType {{.*}} 'float'
// CHECK: | |-ParmVarDecl {{.*}} used z 'double *'
// CHECK: | | `-typeDetails: PointerType {{.*}} 'double *'
// CHECK: | | `-typeDetails: BuiltinType {{.*}} 'double'
// CHECK: | `-CompoundStmt {{.*}} 
// CHECK: | |-IfStmt {{.*}} 
// CHECK: | | |-ImplicitCastExpr {{.*}} 'bool' <PointerToBoolean>
// CHECK: | | | `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK: | | | `-DeclRefExpr {{.*}} 'int *' lvalue ParmVar {{.*}} 'x' 'int *'
// CHECK: | | `-ReturnStmt {{.*}} 
// CHECK: | | `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK: | | `-DeclRefExpr {{.*}} 'int *' lvalue ParmVar {{.*}} 'x' 'int *'
// CHECK: | |-IfStmt {{.*}} 
// CHECK: | | |-ImplicitCastExpr {{.*}} 'bool' <PointerToBoolean>
// CHECK: | | | `-ImplicitCastExpr {{.*}} 'float *' <LValueToRValue>
// CHECK: | | | `-DeclRefExpr {{.*}} 'float *' lvalue ParmVar {{.*}} 'y' 'float *'
// CHECK: | | `-ReturnStmt {{.*}} 
// CHECK: | | `-RecoveryExpr {{.*}} 'int *' contains-errors
// CHECK: | | `-DeclRefExpr {{.*}} 'float *' lvalue ParmVar {{.*}} 'y' 'float *'
// CHECK: | |-IfStmt {{.*}} 
// CHECK: | | |-ImplicitCastExpr {{.*}} 'bool' <PointerToBoolean>
// CHECK: | | | `-ImplicitCastExpr {{.*}} 'double *' <LValueToRValue>
// CHECK: | | | `-DeclRefExpr {{.*}} 'double *' lvalue ParmVar {{.*}} 'z' 'double *'
// CHECK: | | `-ReturnStmt {{.*}} 
// CHECK: | | `-RecoveryExpr {{.*}} 'int *' contains-errors
// CHECK: | | `-DeclRefExpr {{.*}} 'double *' lvalue ParmVar {{.*}} 'z' 'double *'
// CHECK: | `-ReturnStmt {{.*}} 
// CHECK: | `-RecoveryExpr {{.*}} 'int *' contains-errors
// CHECK: | `-DeclRefExpr {{.*}} 'int *' lvalue ParmVar {{.*}} 'x' 'int *'


void returnInitListFromVoid() {
 return {7,8};
}

// CHECK: |-FunctionDecl {{.*}} returnInitListFromVoid 'void ()'
// CHECK: | `-CompoundStmt {{.*}} 
// CHECK: | `-ReturnStmt {{.*}} 
// CHECK: | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | |-IntegerLiteral {{.*}} 'int' 7
// CHECK: | `-IntegerLiteral {{.*}} 'int' 8


void RecoveryExprForInvalidDecls(Unknown InvalidDecl) {
 InvalidDecl + 1;
 InvalidDecl();
}

// CHECK: |-FunctionDecl {{.*}} invalid RecoveryExprForInvalidDecls 'void (int)'
// CHECK: | |-ParmVarDecl {{.*}} referenced invalid InvalidDecl 'int'
// CHECK: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK: | `-CompoundStmt {{.*}} 
// CHECK: | |-BinaryOperator {{.*}} '<dependent type>' contains-errors '+'
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'InvalidDecl' 'int'
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 1
// CHECK: | `-CallExpr {{.*}} '<dependent type>' contains-errors
// CHECK: | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'InvalidDecl' 'int'


void InitializerOfInvalidDecl() {
 int ValidDecl;
 Unkown InvalidDecl = ValidDecl;
 Unknown InvalidDeclWithInvalidInit = Invalid;
}

// CHECK: |-FunctionDecl {{.*}} InitializerOfInvalidDecl 'void ()'
// CHECK: | `-CompoundStmt {{.*}} 
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} used ValidDecl 'int'
// CHECK: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} invalid InvalidDecl 'int' cinit
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'ValidDecl' 'int'
// CHECK: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK: | `-DeclStmt {{.*}} 
// CHECK: | `-VarDecl {{.*}} invalid InvalidDeclWithInvalidInit 'int' cinit
// CHECK: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

void RecoverToAnInvalidDecl() {
 Unknown* foo; // invalid decl
 goo; // the typo was correct to the invalid foo.
 // Verify that RecoveryExpr has an inner DeclRefExpr.
}

// CHECK: |-FunctionDecl {{.*}} RecoverToAnInvalidDecl 'void ()'
// CHECK: | `-CompoundStmt {{.*}} 
// CHECK: | |-DeclStmt {{.*}} 
// CHECK: | | `-VarDecl {{.*}} referenced invalid foo 'int *'
// CHECK: | | `-typeDetails: PointerType {{.*}} 'int *'
// CHECK: | | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK: | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} 'foo' 'int *'

void RecoveryToDoWhileStmtCond() {
 do {} while (some_invalid_val + 1 < 10);
}

void RecoveryForStmtCond() {
 for (int i = 0; i < invalid; ++i) {}
}

// CHECK: |-FunctionDecl {{.*}} RecoveryToDoWhileStmtCond 'void ()'
// CHECK: | `-CompoundStmt {{.*}} 
// CHECK: | `-DoStmt {{.*}} 
// CHECK: | |-CompoundStmt {{.*}} 
// CHECK: | `-BinaryOperator {{.*}} '<dependent type>' contains-errors '<'
// CHECK: | |-BinaryOperator {{.*}} '<dependent type>' contains-errors '+'
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 1
// CHECK: | `-IntegerLiteral {{.*}} 'int' 10
// CHECK: `-FunctionDecl {{.*}} RecoveryForStmtCond 'void ()'
// CHECK: `-CompoundStmt {{.*}} 
// CHECK: `-ForStmt {{.*}} 
// CHECK: |-DeclStmt {{.*}} 
// CHECK: | `-VarDecl {{.*}} used i 'int' cinit
// CHECK: | |-IntegerLiteral {{.*}} 'int' 0
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'
// CHECK: |-<<<NULL>>>
// CHECK: |-RecoveryExpr {{.*}} 'bool' contains-errors
// CHECK: |-UnaryOperator {{.*}} 'int' lvalue prefix '++'
// CHECK: | `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK: `-CompoundStmt {{.*}} 
