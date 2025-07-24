// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++17 \
// RUN: 					 -ast-dump %s -ast-dump-filter test \
// RUN: | FileCheck -strict-whitespace --match-full-lines %s

// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++17 \
// RUN:            -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++17 \
// RUN:            -x c++ -include-pch %t -ast-dump-all -ast-dump-filter test /dev/null \
// RUN: | FileCheck -strict-whitespace --match-full-lines %s



template <typename... Ts> void test(Ts... a) {
  struct V {
    void f() {
      [this] {};
      [*this] {};
    }
  };
  int b, c;
  []() {};
  [](int a, ...) {};
  [a...] {};
  [=] {};
  [=] { return b; };
  [&] {};
  [&] { return c; };
  [b, &c] { return b + c; };
  [a..., x = 12] {};
  []() constexpr {};
  []() mutable {};
  []() noexcept {};
  []() -> int { return 0; };
  [] [[noreturn]] () {};
}


// CHECK:Dumping test:
// CHECK:FunctionTemplateDecl {{.*}} <{{.*}}ast-dump-lambda.cpp:15:1, line:37:1> line:15:32{{( imported)?}} test
// CHECK:|-TemplateTypeParmDecl {{.*}} <col:11, col:23> col:23{{( imported)?}} referenced typename depth 0 index 0 ... Ts
// CHECK:`-FunctionDecl {{.*}} <col:27, line:37:1> line:15:32{{( imported)?}} test 'void (Ts...)'
// CHECK:  |-ParmVarDecl {{.*}} <col:37, col:43> col:43{{( imported)?}} referenced a 'Ts...' pack
// CHECK:  `-CompoundStmt {{.*}} <col:46, line:37:1>
// CHECK:    |-DeclStmt {{.*}} <line:16:3, line:21:4>
// CHECK:    | `-CXXRecordDecl {{.*}} <line:16:3, line:21:3> line:16:10{{( imported)?}}{{( <undeserialized declarations>)?}} struct V definition
// CHECK:    |   |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK:    |   | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// CHECK:    |   | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    |   | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:    |   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    |   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK:    |   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:    |   |-CXXRecordDecl {{.*}} <col:3, col:10> col:10{{( imported)?}} implicit struct V
// CHECK:    |   `-CXXMethodDecl {{.*}} <line:17:5, line:20:5> line:17:10{{( imported)?}} f 'void ()' implicit-inline
// CHECK:    |     `-CompoundStmt {{.*}} <col:14, line:20:5>
// CHECK:    |       |-LambdaExpr {{.*}} <line:18:7, col:15> '(lambda at {{.*}}ast-dump-lambda.cpp:18:7)'
// CHECK:    |       | |-CXXRecordDecl {{.*}} <col:7> col:7{{( imported)?}} implicit{{( <undeserialized declarations>)?}} class definition
// CHECK:    |       | | |-DefinitionData lambda standard_layout trivially_copyable can_const_default_init
// CHECK:    |       | | | |-DefaultConstructor
// CHECK:    |       | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    |       | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:    |       | | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    |       | | | |-MoveAssignment
// CHECK:    |       | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:    |       | | |-CXXMethodDecl {{.*}} <col:12, col:15> col:7{{( imported)?}} operator() 'auto () const -> auto' inline
// CHECK:    |       | | | `-CompoundStmt {{.*}} <col:14, col:15>
// CHECK:    |       | | `-FieldDecl {{.*}} <col:8> col:8{{( imported)?}} implicit 'V *'
// CHECK:    |       | |-ParenListExpr {{.*}} <col:8> 'NULL TYPE'
// CHECK:    |       | | `-CXXThisExpr {{.*}} <col:8> 'V *' this
// CHECK:    |       | `-CompoundStmt {{.*}} <col:14, col:15>
// CHECK:    |       `-LambdaExpr {{.*}} <line:19:7, col:16> '(lambda at {{.*}}ast-dump-lambda.cpp:19:7)'
// CHECK:    |         |-CXXRecordDecl {{.*}} <col:7> col:7{{( imported)?}} implicit{{( <undeserialized declarations>)?}} class definition
// CHECK:    |         | |-DefinitionData lambda standard_layout trivially_copyable can_const_default_init
// CHECK:    |         | | |-DefaultConstructor defaulted_is_constexpr
// CHECK:    |         | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    |         | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:    |         | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    |         | | |-MoveAssignment
// CHECK:    |         | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:    |         | |-CXXMethodDecl {{.*}} <col:13, col:16> col:7{{( imported)?}} operator() 'auto () const -> auto' inline
// CHECK:    |         | | `-CompoundStmt {{.*}} <col:15, col:16>
// CHECK:    |         | `-FieldDecl {{.*}} <col:8> col:8{{( imported)?}} implicit 'V'
// CHECK:    |         |-ParenListExpr {{.*}} <col:8> 'NULL TYPE'
// CHECK:    |         | `-UnaryOperator {{.*}} <col:8> 'V' lvalue prefix '*' cannot overflow
// CHECK:    |         |   `-CXXThisExpr {{.*}} <col:8> 'V *' this
// CHECK:    |         `-CompoundStmt {{.*}} <col:15, col:16>
// CHECK:    |-DeclStmt {{.*}} <line:22:3, col:11>
// CHECK:    | |-VarDecl {{.*}} <col:3, col:7> col:7{{( imported)?}} referenced b 'int'
// CHECK:    | `-VarDecl {{.*}} <col:3, col:10> col:10{{( imported)?}} referenced c 'int'
// CHECK:    |-LambdaExpr {{.*}} <line:23:3, col:9> '(lambda at {{.*}}ast-dump-lambda.cpp:23:3)'
// CHECK:    | |-CXXRecordDecl {{.*}} <col:3> col:3{{( imported)?}} implicit{{( <undeserialized declarations>)?}} class definition
// CHECK:    | | |-DefinitionData lambda empty standard_layout trivially_copyable literal can_const_default_init
// CHECK:    | | | |-DefaultConstructor defaulted_is_constexpr
// CHECK:    | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:    | | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveAssignment
// CHECK:    | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:    | | |-CXXMethodDecl {{.*}} <col:6, col:9> col:3{{( imported)?}} operator() 'auto () const' inline
// CHECK:    | | | `-CompoundStmt {{.*}} <col:8, col:9>
// CHECK:    | | |-CXXConversionDecl {{.*}} <col:3, col:9> col:3{{( imported)?}} implicit constexpr operator auto (*)() 'auto (*() const noexcept)()' inline
// CHECK:    | | `-CXXMethodDecl {{.*}} <col:3, col:9> col:3{{( imported)?}} implicit __invoke 'auto ()' static inline
// CHECK:    | `-CompoundStmt {{.*}} <col:8, col:9>
// CHECK:    |-LambdaExpr {{.*}} <line:24:3, col:19> '(lambda at {{.*}}ast-dump-lambda.cpp:24:3)'
// CHECK:    | |-CXXRecordDecl {{.*}} <col:3> col:3{{( imported)?}} implicit{{( <undeserialized declarations>)?}} class definition
// CHECK:    | | |-DefinitionData lambda empty standard_layout trivially_copyable literal can_const_default_init
// CHECK:    | | | |-DefaultConstructor defaulted_is_constexpr
// CHECK:    | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:    | | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveAssignment
// CHECK:    | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:    | | |-CXXMethodDecl {{.*}} <col:16, col:19> col:3{{( imported)?}} operator() 'auto (int, ...) const' inline
// CHECK:    | | | |-ParmVarDecl {{.*}} <col:6, col:10> col:10{{( imported)?}} a 'int'
// CHECK:    | | | `-CompoundStmt {{.*}} <col:18, col:19>
// CHECK:    | | |-CXXConversionDecl {{.*}} <col:3, col:19> col:3{{( imported)?}} implicit constexpr operator auto (*)(int, ...) 'auto (*() const noexcept)(int, ...)' inline
// CHECK:    | | `-CXXMethodDecl {{.*}} <col:3, col:19> col:3{{( imported)?}} implicit __invoke 'auto (int, ...)' static inline
// CHECK:    | |   `-ParmVarDecl {{.*}} <col:6, col:10> col:10{{( imported)?}} a 'int'
// CHECK:    | `-CompoundStmt {{.*}} <col:18, col:19>
// CHECK:    |-LambdaExpr {{.*}} <line:25:3, col:11> '(lambda at {{.*}}ast-dump-lambda.cpp:25:3)'
// CHECK:    | |-CXXRecordDecl {{.*}} <col:3> col:3{{( imported)?}} implicit{{( <undeserialized declarations>)?}} class definition
// CHECK:    | | |-DefinitionData lambda standard_layout trivially_copyable can_const_default_init
// CHECK:    | | | |-DefaultConstructor
// CHECK:    | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:    | | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveAssignment
// CHECK:    | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:    | | |-CXXMethodDecl {{.*}} <col:8, col:11> col:3{{( imported)?}} operator() 'auto () const -> auto' inline
// CHECK:    | | | `-CompoundStmt {{.*}} <col:10, col:11>
// CHECK:    | | `-FieldDecl {{.*}} <col:4> col:4{{( imported)?}} implicit 'Ts...'
// CHECK:    | |-ParenListExpr {{.*}} <col:4> 'NULL TYPE'
// CHECK:    | | `-DeclRefExpr {{.*}} <col:4> 'Ts' lvalue ParmVar {{.*}} 'a' 'Ts...'
// CHECK:    | `-CompoundStmt {{.*}} <col:10, col:11>
// CHECK:    |-LambdaExpr {{.*}} <line:26:3, col:8> '(lambda at {{.*}}ast-dump-lambda.cpp:26:3)'
// CHECK:    | |-CXXRecordDecl {{.*}} <col:3> col:3{{( imported)?}} implicit{{( <undeserialized declarations>)?}} class definition
// CHECK:    | | |-DefinitionData lambda empty standard_layout trivially_copyable literal can_const_default_init
// CHECK:    | | | |-DefaultConstructor defaulted_is_constexpr
// CHECK:    | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:    | | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveAssignment
// CHECK:    | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:    | | `-CXXMethodDecl {{.*}} <col:5, col:8> col:3{{( imported)?}} operator() 'auto () const -> auto' inline
// CHECK:    | |   `-CompoundStmt {{.*}} <col:7, col:8>
// CHECK:    | `-CompoundStmt {{.*}} <col:7, col:8>
// CHECK:    |-LambdaExpr {{.*}} <line:27:3, col:19> '(lambda at {{.*}}ast-dump-lambda.cpp:27:3)'
// CHECK:    | |-CXXRecordDecl {{.*}} <col:3> col:3{{( imported)?}} implicit{{( <undeserialized declarations>)?}} class definition
// CHECK:    | | |-DefinitionData lambda empty standard_layout trivially_copyable literal can_const_default_init
// CHECK:    | | | |-DefaultConstructor defaulted_is_constexpr
// CHECK:    | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:    | | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveAssignment
// CHECK:    | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:    | | `-CXXMethodDecl {{.*}} <col:5, col:19> col:3{{( imported)?}} operator() 'auto () const -> auto' inline
// CHECK:    | |   `-CompoundStmt {{.*}} <col:7, col:19>
// CHECK:    | |     `-ReturnStmt {{.*}} <col:9, col:16>
// CHECK:    | |       `-DeclRefExpr {{.*}} <col:16> 'const int' lvalue Var {{.*}} 'b' 'int' refers_to_enclosing_variable_or_capture
// CHECK:    | `-CompoundStmt {{.*}} <col:7, col:19>
// CHECK:    |   `-ReturnStmt {{.*}} <col:9, col:16>
// CHECK:    |     `-DeclRefExpr {{.*}} <col:16> 'const int' lvalue Var {{.*}} 'b' 'int' refers_to_enclosing_variable_or_capture
// CHECK:    |-LambdaExpr {{.*}} <line:28:3, col:8> '(lambda at {{.*}}ast-dump-lambda.cpp:28:3)'
// CHECK:    | |-CXXRecordDecl {{.*}} <col:3> col:3{{( imported)?}} implicit{{( <undeserialized declarations>)?}} class definition
// CHECK:    | | |-DefinitionData lambda empty standard_layout trivially_copyable literal can_const_default_init
// CHECK:    | | | |-DefaultConstructor defaulted_is_constexpr
// CHECK:    | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:    | | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveAssignment
// CHECK:    | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:    | | `-CXXMethodDecl {{.*}} <col:5, col:8> col:3{{( imported)?}} operator() 'auto () const -> auto' inline
// CHECK:    | |   `-CompoundStmt {{.*}} <col:7, col:8>
// CHECK:    | `-CompoundStmt {{.*}} <col:7, col:8>
// CHECK:    |-LambdaExpr {{.*}} <line:29:3, col:19> '(lambda at {{.*}}ast-dump-lambda.cpp:29:3)'
// CHECK:    | |-CXXRecordDecl {{.*}} <col:3> col:3{{( imported)?}} implicit{{( <undeserialized declarations>)?}} class definition
// CHECK:    | | |-DefinitionData lambda empty standard_layout trivially_copyable literal can_const_default_init
// CHECK:    | | | |-DefaultConstructor defaulted_is_constexpr
// CHECK:    | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:    | | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveAssignment
// CHECK:    | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:    | | `-CXXMethodDecl {{.*}} <col:5, col:19> col:3{{( imported)?}} operator() 'auto () const -> auto' inline
// CHECK:    | |   `-CompoundStmt {{.*}} <col:7, col:19>
// CHECK:    | |     `-ReturnStmt {{.*}} <col:9, col:16>
// CHECK:    | |       `-DeclRefExpr {{.*}} <col:16> 'int' lvalue Var {{.*}} 'c' 'int' refers_to_enclosing_variable_or_capture
// CHECK:    | `-CompoundStmt {{.*}} <col:7, col:19>
// CHECK:    |   `-ReturnStmt {{.*}} <col:9, col:16>
// CHECK:    |     `-DeclRefExpr {{.*}} <col:16> 'int' lvalue Var {{.*}} 'c' 'int' refers_to_enclosing_variable_or_capture
// CHECK:    |-LambdaExpr {{.*}} <line:30:3, col:27> '(lambda at {{.*}}ast-dump-lambda.cpp:30:3)'
// CHECK:    | |-CXXRecordDecl {{.*}} <col:3> col:3{{( imported)?}} implicit{{( <undeserialized declarations>)?}} class definition
// CHECK:    | | |-DefinitionData lambda trivially_copyable literal can_const_default_init
// CHECK:    | | | |-DefaultConstructor
// CHECK:    | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:    | | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveAssignment
// CHECK:    | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:    | | |-CXXMethodDecl {{.*}} <col:9, col:27> col:3{{( imported)?}} operator() 'auto () const -> auto' inline
// CHECK:    | | | `-CompoundStmt {{.*}} <col:11, col:27>
// CHECK:    | | |   `-ReturnStmt {{.*}} <col:13, col:24>
// CHECK:    | | |     `-BinaryOperator {{.*}} <col:20, col:24> 'int' '+'
// CHECK:    | | |       |-ImplicitCastExpr {{.*}} <col:20> 'int' <LValueToRValue>
// CHECK:    | | |       | `-DeclRefExpr {{.*}} <col:20> 'const int' lvalue Var {{.*}} 'b' 'int' refers_to_enclosing_variable_or_capture
// CHECK:    | | |       `-ImplicitCastExpr {{.*}} <col:24> 'int' <LValueToRValue>
// CHECK:    | | |         `-DeclRefExpr {{.*}} <col:24> 'int' lvalue Var {{.*}} 'c' 'int' refers_to_enclosing_variable_or_capture
// CHECK:    | | |-FieldDecl {{.*}} <col:4> col:4{{( imported)?}} implicit 'int'
// CHECK:    | | `-FieldDecl {{.*}} <col:8> col:8{{( imported)?}} implicit 'int &'
// CHECK:    | |-ImplicitCastExpr {{.*}} <col:4> 'int' <LValueToRValue>
// CHECK:    | | `-DeclRefExpr {{.*}} <col:4> 'int' lvalue Var {{.*}} 'b' 'int'
// CHECK:    | |-DeclRefExpr {{.*}} <col:8> 'int' lvalue Var {{.*}} 'c' 'int'
// CHECK:    | `-CompoundStmt {{.*}} <col:11, col:27>
// CHECK:    |   `-ReturnStmt {{.*}} <col:13, col:24>
// CHECK:    |     `-BinaryOperator {{.*}} <col:20, col:24> 'int' '+'
// CHECK:    |       |-ImplicitCastExpr {{.*}} <col:20> 'int' <LValueToRValue>
// CHECK:    |       | `-DeclRefExpr {{.*}} <col:20> 'const int' lvalue Var {{.*}} 'b' 'int' refers_to_enclosing_variable_or_capture
// CHECK:    |       `-ImplicitCastExpr {{.*}} <col:24> 'int' <LValueToRValue>
// CHECK:    |         `-DeclRefExpr {{.*}} <col:24> 'int' lvalue Var {{.*}} 'c' 'int' refers_to_enclosing_variable_or_capture
// CHECK:    |-LambdaExpr {{.*}} <line:31:3, col:19> '(lambda at {{.*}}ast-dump-lambda.cpp:31:3)'
// CHECK:    | |-CXXRecordDecl {{.*}} <col:3> col:3{{( imported)?}} implicit{{( <undeserialized declarations>)?}} class definition
// CHECK:    | | |-DefinitionData lambda standard_layout trivially_copyable can_const_default_init
// CHECK:    | | | |-DefaultConstructor
// CHECK:    | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:    | | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveAssignment
// CHECK:    | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:    | | |-CXXMethodDecl {{.*}} <col:16, col:19> col:3{{( imported)?}} operator() 'auto () const -> auto' inline
// CHECK:    | | | `-CompoundStmt {{.*}} <col:18, col:19>
// CHECK:    | | |-FieldDecl {{.*}} <col:4> col:4{{( imported)?}} implicit 'Ts...'
// CHECK:    | | `-FieldDecl {{.*}} <col:10> col:10{{( imported)?}} implicit 'int'
// CHECK:    | |-ParenListExpr {{.*}} <col:4> 'NULL TYPE'
// CHECK:    | | `-DeclRefExpr {{.*}} <col:4> 'Ts' lvalue ParmVar {{.*}} 'a' 'Ts...'
// CHECK:    | |-IntegerLiteral {{.*}} <col:14> 'int' 12
// CHECK:    | `-CompoundStmt {{.*}} <col:18, col:19>
// CHECK:    |-LambdaExpr {{.*}} <line:32:3, col:19> '(lambda at {{.*}}ast-dump-lambda.cpp:32:3)'
// CHECK:    | |-CXXRecordDecl {{.*}} <col:3> col:3{{( imported)?}} implicit{{( <undeserialized declarations>)?}} class definition
// CHECK:    | | |-DefinitionData lambda empty standard_layout trivially_copyable literal can_const_default_init
// CHECK:    | | | |-DefaultConstructor defaulted_is_constexpr
// CHECK:    | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:    | | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveAssignment
// CHECK:    | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:    | | |-CXXMethodDecl {{.*}} <col:8, col:19> col:3{{( imported)?}} constexpr operator() 'auto () const' inline
// CHECK:    | | | `-CompoundStmt {{.*}} <col:18, col:19>
// CHECK:    | | |-CXXConversionDecl {{.*}} <col:3, col:19> col:3{{( imported)?}} implicit constexpr operator auto (*)() 'auto (*() const noexcept)()' inline
// CHECK:    | | `-CXXMethodDecl {{.*}} <col:3, col:19> col:3{{( imported)?}} implicit constexpr __invoke 'auto ()' static inline
// CHECK:    | `-CompoundStmt {{.*}} <col:18, col:19>
// CHECK:    |-LambdaExpr {{.*}} <line:33:3, col:17> '(lambda at {{.*}}ast-dump-lambda.cpp:33:3)'
// CHECK:    | |-CXXRecordDecl {{.*}} <col:3> col:3{{( imported)?}} implicit{{( <undeserialized declarations>)?}} class definition
// CHECK:    | | |-DefinitionData lambda empty standard_layout trivially_copyable literal can_const_default_init
// CHECK:    | | | |-DefaultConstructor defaulted_is_constexpr
// CHECK:    | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:    | | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveAssignment
// CHECK:    | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:    | | |-CXXMethodDecl {{.*}} <col:8, col:17> col:3{{( imported)?}} operator() 'auto ()' inline
// CHECK:    | | | `-CompoundStmt {{.*}} <col:16, col:17>
// CHECK:    | | |-CXXConversionDecl {{.*}} <col:3, col:17> col:3{{( imported)?}} implicit constexpr operator auto (*)() 'auto (*() const noexcept)()' inline
// CHECK:    | | `-CXXMethodDecl {{.*}} <col:3, col:17> col:3{{( imported)?}} implicit __invoke 'auto ()' static inline
// CHECK:    | `-CompoundStmt {{.*}} <col:16, col:17>
// CHECK:    |-LambdaExpr {{.*}} <line:34:3, col:18> '(lambda at {{.*}}ast-dump-lambda.cpp:34:3)'
// CHECK:    | |-CXXRecordDecl {{.*}} <col:3> col:3{{( imported)?}} implicit{{( <undeserialized declarations>)?}} class definition
// CHECK:    | | |-DefinitionData lambda empty standard_layout trivially_copyable literal can_const_default_init
// CHECK:    | | | |-DefaultConstructor defaulted_is_constexpr
// CHECK:    | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:    | | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveAssignment
// CHECK:    | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:    | | |-CXXMethodDecl {{.*}} <col:8, col:18> col:3{{( imported)?}} operator() 'auto () const noexcept' inline
// CHECK:    | | | `-CompoundStmt {{.*}} <col:17, col:18>
// CHECK:    | | |-CXXConversionDecl {{.*}} <col:3, col:18> col:3{{( imported)?}} implicit constexpr operator auto (*)() noexcept 'auto (*() const noexcept)() noexcept' inline
// CHECK:    | | `-CXXMethodDecl {{.*}} <col:3, col:18> col:3{{( imported)?}} implicit __invoke 'auto () noexcept' static inline
// CHECK:    | `-CompoundStmt {{.*}} <col:17, col:18>
// CHECK:    |-LambdaExpr {{.*}} <line:35:3, col:27> '(lambda at {{.*}}ast-dump-lambda.cpp:35:3)'
// CHECK:    | |-CXXRecordDecl {{.*}} <col:3> col:3{{( imported)?}} implicit{{( <undeserialized declarations>)?}} class definition
// CHECK:    | | |-DefinitionData lambda empty standard_layout trivially_copyable literal can_const_default_init
// CHECK:    | | | |-DefaultConstructor defaulted_is_constexpr
// CHECK:    | | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:    | | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:    | | | |-MoveAssignment
// CHECK:    | | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:    | | |-CXXMethodDecl {{.*}} <col:11, col:27> col:3{{( imported)?}} operator() 'auto () const -> int' inline
// CHECK:    | | | `-CompoundStmt {{.*}} <col:15, col:27>
// CHECK:    | | |   `-ReturnStmt {{.*}} <col:17, col:24>
// CHECK:    | | |     `-IntegerLiteral {{.*}} <col:24> 'int' 0
// CHECK:    | | |-CXXConversionDecl {{.*}} <col:3, col:27> col:3{{( imported)?}} implicit constexpr operator int (*)() 'auto (*() const noexcept)() -> int' inline
// CHECK:    | | `-CXXMethodDecl {{.*}} <col:3, col:27> col:3{{( imported)?}} implicit __invoke 'auto () -> int' static inline
// CHECK:    | `-CompoundStmt {{.*}} <col:15, col:27>
// CHECK:    |   `-ReturnStmt {{.*}} <col:17, col:24>
// CHECK:    |     `-IntegerLiteral {{.*}} <col:24> 'int' 0
// CHECK:    `-LambdaExpr {{.*}} <line:36:3, col:23> '(lambda at {{.*}}ast-dump-lambda.cpp:36:3)'
// CHECK:      |-CXXRecordDecl {{.*}} <col:3> col:3{{( imported)?}} implicit{{( <undeserialized declarations>)?}} class definition
// CHECK:      | |-DefinitionData lambda empty standard_layout trivially_copyable literal can_const_default_init
// CHECK:      | | |-DefaultConstructor defaulted_is_constexpr
// CHECK:      | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:      | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK:      | | |-CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
// CHECK:      | | |-MoveAssignment
// CHECK:      | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK:      | |-CXXMethodDecl {{.*}} <col:20, col:23> col:3{{( imported)?}} operator() 'auto () const' inline
// CHECK:      | | |-CompoundStmt {{.*}} <col:22, col:23>
// CHECK:      | | `-attrDetails: CXX11NoReturnAttr {{.*}} <col:8> noreturn
// CHECK:      | |-CXXConversionDecl {{.*}} <col:3, col:23> col:3{{( imported)?}} implicit constexpr operator auto (*)() 'auto (*() const noexcept)()' inline
// CHECK:      | `-CXXMethodDecl {{.*}} <col:3, col:23> col:3{{( imported)?}} implicit __invoke 'auto ()' static inline
// CHECK:      `-CompoundStmt {{.*}} <col:22, col:23>
