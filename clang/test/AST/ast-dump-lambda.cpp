// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++23 -ast-dump      -ast-dump-filter test %s | FileCheck --match-full-lines --check-prefix DUMP %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++23 -ast-dump=json -ast-dump-filter test %s | FileCheck --match-full-lines --check-prefix JSON %s

// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++23 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++23 -x c++ -include-pch %t -ast-dump-all                -ast-dump-filter test /dev/null | FileCheck --match-full-lines --check-prefix DUMP %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++23 -x c++ -include-pch %t -ast-dump-all -ast-dump=json -ast-dump-filter test /dev/null | FileCheck --match-full-lines --check-prefix JSON %s


template <typename... Ts> void test(Ts... a) {
  struct V {
    void f() {
      auto case1 = [this] {};
      auto case2 = [*this] {};
    }
  };
  double b, c;
  auto caseA = []() {};
  auto caseB = [](void) {};
  auto caseC = [] {};
  auto caseD = [](int a, ...) {};
  auto caseE = [a...] {};
  auto caseF = [=] () {};
  auto caseG = [=] { return b; };
  auto caseH = [&] () {};
  auto caseI = [&] { return c; };
  auto caseJ = [b, &c] { return b + c; };
  auto caseK = [b,c](){};
  auto caseL = [a..., x = 12] {};
  auto caseM = []() constexpr {};
  auto caseN = []() mutable {};
  auto caseO = []() noexcept {};
  auto caseP = []() -> int { return 0; };
  auto caseQ = [] -> int { return 0; };
  auto caseR = [] [[noreturn]] () {};
}

// DUMP-LABEL:     |       | `-VarDecl {{.*}} case1 'auto' cinit
// DUMP-NEXT:      |       |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |       |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |       |     | |-DefinitionData lambda standard_layout trivially_copyable can_const_default_init
// DUMP-NEXT:      |       |     | | |-DefaultConstructor
// DUMP-NEXT:      |       |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |       |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |       |     | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |       |     | | |-MoveAssignment
// DUMP-NEXT:      |       |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |       |     | |-CXXMethodDecl {{.*}} operator() 'auto () const -> auto' inline
// DUMP-NEXT:      |       |     | | `-CompoundStmt {{.*}}
// DUMP-NEXT:      |       |     | `-FieldDecl {{.*}} implicit 'V *'
// DUMP-NEXT:      |       |     |-ParenListExpr {{.*}} 'NULL TYPE'
// DUMP-NEXT:      |       |     | `-CXXThisExpr {{.*}} 'V *' this
// DUMP-NEXT:      |       |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |       `-DeclStmt {{.*}}
// DUMP-LABEL:     |         `-VarDecl {{.*}} case2 'auto' cinit
// DUMP-NEXT:      |           `-LambdaExpr {{.*}}
// DUMP-NEXT:      |             |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |             | |-DefinitionData lambda standard_layout trivially_copyable can_const_default_init
// DUMP-NEXT:      |             | | |-DefaultConstructor defaulted_is_constexpr
// DUMP-NEXT:      |             | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |             | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |             | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |             | | |-MoveAssignment
// DUMP-NEXT:      |             | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |             | |-CXXMethodDecl {{.*}} operator() 'auto () const -> auto' inline
// DUMP-NEXT:      |             | | `-CompoundStmt {{.*}}
// DUMP-NEXT:      |             | `-FieldDecl {{.*}} implicit 'V'
// DUMP-NEXT:      |             |-ParenListExpr {{.*}} 'NULL TYPE'
// DUMP-NEXT:      |             | `-UnaryOperator {{.*}} 'V' lvalue prefix '*' cannot overflow
// DUMP-NEXT:      |             |   `-CXXThisExpr {{.*}} 'V *' this
// DUMP-NEXT:      |             `-CompoundStmt {{.*}}
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-NEXT:      | |-VarDecl {{.*}} referenced b 'double'
// DUMP-NEXT:      | `-VarDecl {{.*}} referenced c 'double'
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:     | `-VarDecl {{.*}} caseA 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda empty standard_layout trivially_copyable trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | |-CXXMethodDecl {{.*}} operator() 'auto () const' inline
// DUMP-NEXT:      |     | | `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     | |-CXXConversionDecl {{.*}} implicit constexpr operator auto (*)() 'auto (*() const noexcept)()' inline
// DUMP-NEXT:      |     | `-CXXMethodDecl {{.*}} implicit __invoke 'auto ()' static inline
// DUMP-NEXT:      |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:      | `-VarDecl {{.*}} caseB 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda empty standard_layout trivially_copyable trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | |-CXXMethodDecl {{.*}} operator() 'auto () const' inline
// DUMP-NEXT:      |     | | `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     | |-CXXConversionDecl {{.*}} implicit constexpr operator auto (*)() 'auto (*() const noexcept)()' inline
// DUMP-NEXT:      |     | `-CXXMethodDecl {{.*}} implicit __invoke 'auto ()' static inline
// DUMP-NEXT:      |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:     | `-VarDecl {{.*}} caseC 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda empty standard_layout trivially_copyable trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | |-CXXMethodDecl {{.*}} operator() 'auto () const -> auto' inline
// DUMP-NEXT:      |     | | `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     | |-CXXConversionDecl {{.*}} implicit constexpr operator auto (*)() 'auto (*() const noexcept)() -> auto' inline
// DUMP-NEXT:      |     | `-CXXMethodDecl {{.*}} implicit __invoke 'auto () -> auto' static inline
// DUMP-NEXT:      |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:     | `-VarDecl {{.*}} caseD 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda empty standard_layout trivially_copyable trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | |-CXXMethodDecl {{.*}} operator() 'auto (int, ...) const' inline
// DUMP-NEXT:      |     | | |-ParmVarDecl {{.*}} a 'int'
// DUMP-NEXT:      |     | | `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     | |-CXXConversionDecl {{.*}} implicit constexpr operator auto (*)(int, ...) 'auto (*() const noexcept)(int, ...)' inline
// DUMP-NEXT:      |     | `-CXXMethodDecl {{.*}} implicit __invoke 'auto (int, ...)' static inline
// DUMP-NEXT:      |     |   `-ParmVarDecl {{.*}} a 'int'
// DUMP-NEXT:      |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:     | `-VarDecl {{.*}} caseE 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda standard_layout trivially_copyable can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | |-CXXMethodDecl {{.*}} operator() 'auto () const -> auto' inline
// DUMP-NEXT:      |     | | `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     | `-FieldDecl {{.*}} implicit 'Ts...'
// DUMP-NEXT:      |     |-ParenListExpr {{.*}} 'NULL TYPE'
// DUMP-NEXT:      |     | `-DeclRefExpr {{.*}} 'Ts' lvalue ParmVar 0x{{.*}} 'a' 'Ts...'
// DUMP-NEXT:      |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:     | `-VarDecl {{.*}} caseF 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda empty standard_layout trivially_copyable literal can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor defaulted_is_constexpr
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | `-CXXMethodDecl {{.*}} operator() 'auto () const' inline
// DUMP-NEXT:      |     |   `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:     | `-VarDecl {{.*}} caseG 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda empty standard_layout trivially_copyable literal can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor defaulted_is_constexpr
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | `-CXXMethodDecl {{.*}} operator() 'auto () const -> auto' inline
// DUMP-NEXT:      |     |   `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     |     `-ReturnStmt {{.*}}
// DUMP-NEXT:      |     |       `-DeclRefExpr {{.*}} 'const double' lvalue Var 0x{{.*}} 'b' 'double' refers_to_enclosing_variable_or_capture
// DUMP-NEXT:      |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |       `-ReturnStmt {{.*}}
// DUMP-NEXT:      |         `-DeclRefExpr {{.*}} 'const double' lvalue Var {{.*}} 'b' 'double' refers_to_enclosing_variable_or_capture
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:     | `-VarDecl {{.*}} caseH 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda empty standard_layout trivially_copyable literal can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor defaulted_is_constexpr
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | `-CXXMethodDecl {{.*}} operator() 'auto () const' inline
// DUMP-NEXT:      |     |   `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:     | `-VarDecl {{.*}} caseI 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda empty standard_layout trivially_copyable literal can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor defaulted_is_constexpr
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | `-CXXMethodDecl {{.*}} operator() 'auto () const -> auto' inline
// DUMP-NEXT:      |     |   `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     |     `-ReturnStmt {{.*}}
// DUMP-NEXT:      |     |       `-DeclRefExpr {{.*}} 'double' lvalue Var 0x{{.*}} 'c' 'double' refers_to_enclosing_variable_or_capture
// DUMP-NEXT:      |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |       `-ReturnStmt {{.*}}
// DUMP-NEXT:      |         `-DeclRefExpr {{.*}} 'double' lvalue Var 0x{{.*}} 'c' 'double' refers_to_enclosing_variable_or_capture
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:     | `-VarDecl {{.*}} caseJ 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda trivially_copyable literal can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor defaulted_is_constexpr
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | |-CXXMethodDecl {{.*}} operator() 'auto () const -> auto' inline
// DUMP-NEXT:      |     | | `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     | |   `-ReturnStmt {{.*}}
// DUMP-NEXT:      |     | |     `-BinaryOperator {{.*}} 'double' '+'
// DUMP-NEXT:      |     | |       |-ImplicitCastExpr {{.*}} 'double' <LValueToRValue>
// DUMP-NEXT:      |     | |       | `-DeclRefExpr {{.*}} 'const double' lvalue Var 0x{{.*}} 'b' 'double' refers_to_enclosing_variable_or_capture
// DUMP-NEXT:      |     | |       `-ImplicitCastExpr {{.*}} 'double' <LValueToRValue>
// DUMP-NEXT:      |     | |         `-DeclRefExpr {{.*}} 'double' lvalue Var 0x{{.*}} 'c' 'double' refers_to_enclosing_variable_or_capture
// DUMP-NEXT:      |     | |-FieldDecl {{.*}} implicit 'double'
// DUMP-NEXT:      |     | `-FieldDecl {{.*}} implicit 'double &'
// DUMP-NEXT:      |     |-ImplicitCastExpr {{.*}} 'double' <LValueToRValue>
// DUMP-NEXT:      |     | `-DeclRefExpr {{.*}} 'double' lvalue Var 0x{{.*}} 'b' 'double'
// DUMP-NEXT:      |     |-DeclRefExpr {{.*}} 'double' lvalue Var 0x{{.*}} 'c' 'double'
// DUMP-NEXT:      |     `-CompoundStmt{{.*}}
// DUMP-NEXT:      |       `-ReturnStmt {{.*}}
// DUMP-NEXT:      |         `-BinaryOperator {{.*}} 'double' '+'
// DUMP-NEXT:      |           |-ImplicitCastExpr {{.*}} 'double' <LValueToRValue>
// DUMP-NEXT:      |           | `-DeclRefExpr {{.*}} 'const double' lvalue Var 0x{{.*}} 'b' 'double' refers_to_enclosing_variable_or_capture
// DUMP-NEXT:      |           `-ImplicitCastExpr {{.*}} 'double' <LValueToRValue>
// DUMP-NEXT:      |             `-DeclRefExpr {{.*}} 'double' lvalue Var 0x{{.*}} 'c' 'double' refers_to_enclosing_variable_or_capture
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:     | `-VarDecl {{.*}} caseK 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda standard_layout trivially_copyable literal can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor defaulted_is_constexpr 
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | |-CXXMethodDecl {{.*}} operator() 'auto () const' inline
// DUMP-NEXT:      |     | | `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     | |-FieldDecl {{.*}} implicit 'double'
// DUMP-NEXT:      |     | `-FieldDecl {{.*}} implicit 'double'
// DUMP-NEXT:      |     |-ImplicitCastExpr {{.*}} 'double' <LValueToRValue>
// DUMP-NEXT:      |     | `-DeclRefExpr {{.*}} 'double' lvalue Var 0x{{.*}} 'b' 'double'
// DUMP-NEXT:      |     |-ImplicitCastExpr {{.*}} 'double' <LValueToRValue>
// DUMP-NEXT:      |     | `-DeclRefExpr {{.*}} 'double' lvalue Var 0x{{.*}} 'c' 'double'
// DUMP-NEXT:      |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:     | `-VarDecl {{.*}} caseL 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda standard_layout trivially_copyable can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | |-CXXMethodDecl {{.*}} operator() 'auto () const -> auto' inline
// DUMP-NEXT:      |     | | `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     | |-FieldDecl {{.*}} implicit 'Ts...'
// DUMP-NEXT:      |     | `-FieldDecl {{.*}} implicit 'int'
// DUMP-NEXT:      |     |-ParenListExpr {{.*}} 'NULL TYPE'
// DUMP-NEXT:      |     | `-DeclRefExpr {{.*}} 'Ts' lvalue ParmVar 0x{{.*}} 'a' 'Ts...'
// DUMP-NEXT:      |     |-IntegerLiteral {{.*}} 'int' 12
// DUMP-NEXT:      |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:     | `-VarDecl {{.*}} caseM 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda empty standard_layout trivially_copyable trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | |-CXXMethodDecl {{.*}} constexpr operator() 'auto () const' inline
// DUMP-NEXT:      |     | | `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     | |-CXXConversionDecl {{.*}} implicit constexpr operator auto (*)() 'auto (*() const noexcept)()' inline
// DUMP-NEXT:      |     | `-CXXMethodDecl {{.*}} implicit constexpr __invoke 'auto ()' static inline
// DUMP-NEXT:      |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:     | `-VarDecl {{.*}} caseN 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda empty standard_layout trivially_copyable trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | |-CXXMethodDecl {{.*}} operator() 'auto ()' inline
// DUMP-NEXT:      |     | | `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     | |-CXXConversionDecl {{.*}} implicit constexpr operator auto (*)() 'auto (*() const noexcept)()' inline
// DUMP-NEXT:      |     | `-CXXMethodDecl {{.*}} implicit __invoke 'auto ()' static inline
// DUMP-NEXT:      |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:     | `-VarDecl {{.*}} caseO 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda empty standard_layout trivially_copyable trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | |-CXXMethodDecl {{.*}} operator() 'auto () const noexcept' inline
// DUMP-NEXT:      |     | | `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     | |-CXXConversionDecl {{.*}} implicit constexpr operator auto (*)() noexcept 'auto (*() const noexcept)() noexcept' inline
// DUMP-NEXT:      |     | `-CXXMethodDecl {{.*}} implicit __invoke 'auto () noexcept' static inline
// DUMP-NEXT:      |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:     | `-VarDecl {{.*}} caseP 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda empty standard_layout trivially_copyable trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | |-CXXMethodDecl {{.*}} operator() 'auto () const -> int' inline
// DUMP-NEXT:      |     | | `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     | |   `-ReturnStmt {{.*}}
// DUMP-NEXT:      |     | |     `-IntegerLiteral {{.*}} 'int' 0
// DUMP-NEXT:      |     | |-CXXConversionDecl {{.*}} implicit constexpr operator int (*)() 'auto (*() const noexcept)() -> int' inline
// DUMP-NEXT:      |     | `-CXXMethodDecl {{.*}} implicit __invoke 'auto () -> int' static inline
// DUMP-NEXT:      |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |       `-ReturnStmt {{.*}}
// DUMP-NEXT:      |         `-IntegerLiteral {{.*}} 'int' 0
// DUMP-NEXT:      |-DeclStmt {{.*}}
// DUMP-LABEL:     | `-VarDecl {{.*}} caseQ 'auto' cinit
// DUMP-NEXT:      |   `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:      |     |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:      |     | |-DefinitionData lambda empty standard_layout trivially_copyable trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// DUMP-NEXT:      |     | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// DUMP-NEXT:      |     | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:      |     | | |-MoveAssignment exists simple trivial needs_implicit
// DUMP-NEXT:      |     | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:      |     | |-CXXMethodDecl {{.*}} operator() 'auto () const -> int' inline
// DUMP-NEXT:      |     | | `-CompoundStmt {{.*}}
// DUMP-NEXT:      |     | |   `-ReturnStmt {{.*}}
// DUMP-NEXT:      |     | |     `-IntegerLiteral {{.*}} 'int' 0
// DUMP-NEXT:      |     | |-CXXConversionDecl {{.*}} implicit constexpr operator int (*)() 'auto (*() const noexcept)() -> int' inline
// DUMP-NEXT:      |     | `-CXXMethodDecl {{.*}} implicit __invoke 'auto () -> int' static inline
// DUMP-NEXT:      |     `-CompoundStmt {{.*}}
// DUMP-NEXT:      |       `-ReturnStmt {{.*}}
// DUMP-NEXT:      |         `-IntegerLiteral {{.*}} 'int' 0
// DUMP-NEXT:      `-DeclStmt {{.*}}
// DUMP-LABEL:       `-VarDecl {{.*}} caseR 'auto' cinit
// DUMP-NEXT:          `-LambdaExpr {{.*}} '(lambda at {{.*}})'
// DUMP-NEXT:            |-CXXRecordDecl {{.*}} implicit{{( <undeserialized declarations>)?}} class definition
// DUMP-NEXT:            | |-DefinitionData lambda empty standard_layout trivially_copyable trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// DUMP-NEXT:            | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// DUMP-NEXT:            | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:            | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:            | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:            | | |-MoveAssignment exists simple trivial needs_implicit
// DUMP-NEXT:            | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:            | |-CXXMethodDecl {{.*}} operator() 'auto () const' inline
// DUMP-NEXT:            | | |-CompoundStmt {{.*}}
// DUMP-NEXT:            | | `-CXX11NoReturnAttr {{.*}} noreturn
// DUMP-NEXT:            | |-CXXConversionDecl {{.*}} implicit constexpr operator auto (*)() 'auto (*() const noexcept)()' inline
// DUMP-NEXT:            | `-CXXMethodDecl {{.*}} implicit __invoke 'auto ()' static inline
// DUMP-NEXT:            `-CompoundStmt {{.*}}

// FIXME-LABEL: "name": "case1",
// FIXME:           "hasExplicitParameters": false,
// FIXME-LABEL: "name": "case2",
// FIXME:           "hasExplicitParameters": false,
// JSON-LABEL: "name": "caseA",
// JSON:           "hasExplicitParameters": true,
// JSON-LABEL: "name": "caseB",
// JSON:           "hasExplicitParameters": true,
// JSON-LABEL: "name": "caseC",
// JSON:           "hasExplicitParameters": false,
// JSON-LABEL: "name": "caseD",
// JSON:           "hasExplicitParameters": true,
// JSON-LABEL: "name": "caseE",
// JSON:           "hasExplicitParameters": false,
// JSON-LABEL: "name": "caseF",
// JSON:           "hasExplicitParameters": true,
// JSON-LABEL: "name": "caseG",
// JSON:           "hasExplicitParameters": false,
// JSON-LABEL: "name": "caseH",
// JSON:           "hasExplicitParameters": true,
// JSON-LABEL: "name": "caseI",
// JSON:           "hasExplicitParameters": false,
// JSON-LABEL: "name": "caseJ",
// JSON:           "hasExplicitParameters": false,
// JSON-LABEL: "name": "caseK",
// JSON:           "hasExplicitParameters": true,
// JSON-LABEL: "name": "caseL",
// JSON:           "hasExplicitParameters": false,
// JSON-LABEL: "name": "caseM",
// JSON:           "hasExplicitParameters": true,
// JSON-LABEL: "name": "caseN",
// JSON:           "hasExplicitParameters": true,
// JSON-LABEL: "name": "caseO",
// JSON:           "hasExplicitParameters": true,
// JSON-LABEL: "name": "caseP",
// JSON:           "hasExplicitParameters": true,
// JSON-LABEL: "name": "caseQ",
// JSON:           "hasExplicitParameters": false,
// JSON-LABEL: "name": "caseR",
// JSON:           "hasExplicitParameters": true,
