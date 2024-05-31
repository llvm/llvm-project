// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -fcxx-exceptions -std=gnu++17 -ast-dump %s \
// RUN: | FileCheck --strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -fcxx-exceptions -std=gnu++17 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-unknown -Wno-unused-value -fcxx-exceptions -std=gnu++17 \
// RUN: -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

namespace std {
using size_t = decltype(sizeof(0));

class type_info {
public:
  virtual ~type_info();
  bool operator==(const type_info& rhs) const noexcept;
  bool operator!=(const type_info& rhs) const noexcept;
  type_info(const type_info& rhs) = delete; // cannot be copied
  type_info& operator=(const type_info& rhs) = delete; // cannot be copied
};

class bad_typeid {
public:
  bad_typeid() noexcept;
  bad_typeid(const bad_typeid&) noexcept;
  virtual ~bad_typeid();
  bad_typeid& operator=(const bad_typeid&) noexcept;
  const char* what() const noexcept;
};
} // namespace std
void *operator new(std::size_t, void *ptr);

struct S {
  virtual ~S() = default;

  void func(int);
  template <typename Ty>
  Ty foo();

  int i;
};

struct T : S {};

template <typename>
struct U {};

void Throw() {
  throw 12;
  // CHECK: CXXThrowExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:9> 'void'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:9> 'int' 12

  throw;
  // CHECK: CXXThrowExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3> 'void'
}

void PointerToMember(S obj1, S *obj2, int S::* data, void (S::*call)(int)) {
  obj1.*data;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:9> 'int' lvalue '.*'
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'S' lvalue ParmVar 0x{{[^ ]*}} 'obj1' 'S'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:9> 'int S::*' lvalue ParmVar 0x{{[^ ]*}} 'data' 'int S::*'

  obj2->*data;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:10> 'int' lvalue '->*'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'S *' lvalue ParmVar 0x{{[^ ]*}} 'obj2' 'S *'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:10> 'int S::*' lvalue ParmVar 0x{{[^ ]*}} 'data' 'int S::*'

  (obj1.*call)(12);
  // CHECK: CXXMemberCallExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:18> 'void'
  // CHECK-NEXT: ParenExpr 0x{{[^ ]*}} <col:3, col:14> '<bound member function type>'
  // CHECK-NEXT: BinaryOperator 0x{{[^ ]*}} <col:4, col:10> '<bound member function type>' '.*'
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:4> 'S' lvalue ParmVar 0x{{[^ ]*}} 'obj1' 'S'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:10> 'void (S::*)(int)' lvalue ParmVar 0x{{[^ ]*}} 'call' 'void (S::*)(int)'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:16> 'int' 12

  (obj2->*call)(12);
  // CHECK: CXXMemberCallExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:19> 'void'
  // CHECK-NEXT: ParenExpr 0x{{[^ ]*}} <col:3, col:15> '<bound member function type>'
  // CHECK-NEXT: BinaryOperator 0x{{[^ ]*}} <col:4, col:11> '<bound member function type>' '->*'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:4> 'S *' lvalue ParmVar 0x{{[^ ]*}} 'obj2' 'S *'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:11> 'void (S::*)(int)' lvalue ParmVar 0x{{[^ ]*}} 'call' 'void (S::*)(int)'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:17> 'int' 12
}

void Casting(const S *s) {
  const_cast<S *>(s);
  // CHECK: CXXConstCastExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:20> 'S *' const_cast<S *> <NoOp>
  // CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:19> 'const S *' <LValueToRValue> part_of_explicit_cast
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:19> 'const S *' lvalue ParmVar 0x{{[^ ]*}} 's' 'const S *'

  static_cast<const T *>(s);
  // CHECK: CXXStaticCastExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:27> 'const T *' static_cast<const T *> <BaseToDerived (S)>
  // CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:26> 'const S *' <LValueToRValue> part_of_explicit_cast
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:26> 'const S *' lvalue ParmVar 0x{{[^ ]*}} 's' 'const S *'

  dynamic_cast<const T *>(s);
  // CHECK: CXXDynamicCastExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:28> 'const T *' dynamic_cast<const T *> <Dynamic>
  // CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:27> 'const S *' <LValueToRValue> part_of_explicit_cast
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:27> 'const S *' lvalue ParmVar 0x{{[^ ]*}} 's' 'const S *'

  reinterpret_cast<const int *>(s);
  // CHECK: CXXReinterpretCastExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:34> 'const int *' reinterpret_cast<const int *> <BitCast>
  // CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}} <col:33> 'const S *' <LValueToRValue> part_of_explicit_cast
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:33> 'const S *' lvalue ParmVar 0x{{[^ ]*}} 's' 'const S *'
}

template <typename... Ts>
void UnaryExpressions(int *p) {
  sizeof...(Ts);
  // CHECK: SizeOfPackExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:15> 'unsigned long' 0x{{[^ ]*}} Ts

  noexcept(p - p);
  // CHECK: CXXNoexceptExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:17> 'bool'
  // CHECK-NEXT: BinaryOperator 0x{{[^ ]*}} <col:12, col:16> 'long' '-'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:12> 'int *' lvalue ParmVar 0x{{[^ ]*}} 'p' 'int *'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:16> 'int *' lvalue ParmVar 0x{{[^ ]*}} 'p' 'int *'

  ::new int;
  // CHECK: CXXNewExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:9> 'int *' global Function 0x{{[^ ]*}} 'operator new' 'void *(unsigned long)'

  new (int);
  // CHECK: CXXNewExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:11> 'int *' Function 0x{{[^ ]*}} 'operator new' 'void *(unsigned long)'

  new int{12};
  // CHECK: CXXNewExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:13> 'int *' Function 0x{{[^ ]*}} 'operator new' 'void *(unsigned long)'
  // CHECK-NEXT: InitListExpr 0x{{[^ ]*}} <col:10, col:13> 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:11> 'int' 12

  new int[2];
  // CHECK: CXXNewExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:12> 'int *' array Function 0x{{[^ ]*}} 'operator new[]' 'void *(unsigned long)'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:11> 'int' 2

  new int[2]{1, 2};
  // CHECK: CXXNewExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:18> 'int *' array Function 0x{{[^ ]*}} 'operator new[]' 'void *(unsigned long)'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:11> 'int' 2
  // CHECK-NEXT: InitListExpr 0x{{[^ ]*}} <col:13, col:18> 'int[2]'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:14> 'int' 1
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:17> 'int' 2

  new (p) int;
  // CHECK: CXXNewExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:11> 'int *' Function 0x{{[^ ]*}} 'operator new' 'void *(std::size_t, void *)'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void *' <BitCast>
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:8> 'int *' lvalue ParmVar 0x{{[^ ]*}} 'p' 'int *'

  new (p) int{12};
  // CHECK: CXXNewExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:17> 'int *' Function 0x{{[^ ]*}} 'operator new' 'void *(std::size_t, void *)'
  // CHECK-NEXT: InitListExpr 0x{{[^ ]*}} <col:14, col:17> 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:15> 'int' 12
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'void *' <BitCast>
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:8> 'int *' lvalue ParmVar 0x{{[^ ]*}} 'p' 'int *'

  ::delete p;
  // CHECK: CXXDeleteExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:12> 'void' global Function 0x{{[^ ]*}} 'operator delete' 'void (void *, unsigned long) noexcept'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:12> 'int *' lvalue ParmVar 0x{{[^ ]*}} 'p' 'int *'

  delete [] p;
  // CHECK: CXXDeleteExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:13> 'void' array Function 0x{{[^ ]*}} 'operator delete[]' 'void (void *) noexcept'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:13> 'int *' lvalue ParmVar 0x{{[^ ]*}} 'p' 'int *'
}

void PostfixExpressions(S a, S *p, U<int> *r) {
  a.func(0);
  // CHECK: CXXMemberCallExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:11> 'void'
  // CHECK-NEXT: MemberExpr 0x{{[^ ]*}} <col:3, col:5> '<bound member function type>' .func 0x{{[^ ]*}}
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'S' lvalue ParmVar 0x{{[^ ]*}} 'a' 'S'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:10> 'int' 0

  p->func(0);
  // CHECK: CXXMemberCallExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:12> 'void'
  // CHECK-NEXT: MemberExpr 0x{{[^ ]*}} <col:3, col:6> '<bound member function type>' ->func 0x{{[^ ]*}}
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'S *' lvalue ParmVar 0x{{[^ ]*}} 'p' 'S *'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:11> 'int' 0

  // FIXME: there is no mention that this used the template keyword.
  p->template foo<int>();
  // CHECK: CXXMemberCallExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:24> 'int'
  // CHECK-NEXT: MemberExpr 0x{{[^ ]*}} <col:3, col:22> '<bound member function type>' ->foo 0x{{[^ ]*}}
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'S *' lvalue ParmVar 0x{{[^ ]*}} 'p' 'S *'

  // FIXME: there is no mention that this used the template keyword.
  a.template foo<float>();
  // CHECK: CXXMemberCallExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:25> 'float'
  // CHECK-NEXT: MemberExpr 0x{{[^ ]*}} <col:3, col:23> '<bound member function type>' .foo 0x{{[^ ]*}}
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'S' lvalue ParmVar 0x{{[^ ]*}} 'a' 'S'

  p->~S();
  // CHECK: CXXMemberCallExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:9> 'void'
  // CHECK-NEXT: MemberExpr 0x{{[^ ]*}} <col:3, col:7> '<bound member function type>' ->~S 0x{{[^ ]*}}
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'S *' lvalue ParmVar 0x{{[^ ]*}} 'p' 'S *'

  a.~S();
  // CHECK: CXXMemberCallExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:8> 'void'
  // CHECK-NEXT: MemberExpr 0x{{[^ ]*}} <col:3, col:6> '<bound member function type>' .~S 0x{{[^ ]*}}
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'S' lvalue ParmVar 0x{{[^ ]*}} 'a' 'S'

  // FIXME: there seems to be no way to distinguish the construct below from
  // the construct above.
  a.~decltype(a)();
  // CHECK: CXXMemberCallExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:18> 'void'
  // CHECK-NEXT: MemberExpr 0x{{[^ ]*}} <col:3, col:5> '<bound member function type>' .~S 0x{{[^ ]*}}
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'S' lvalue ParmVar 0x{{[^ ]*}} 'a' 'S'

  // FIXME: similarly, there is no way to distinguish the construct below from
  // the p->~S() case.
  p->::S::~S();
  // CHECK: CXXMemberCallExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:14> 'void'
  // CHECK-NEXT: MemberExpr 0x{{[^ ]*}} <col:3, col:12> '<bound member function type>' ->~S 0x{{[^ ]*}}
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'S'
  // CHECK-NEXT: NestedNameSpecifier Global
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'S *' lvalue ParmVar 0x{{[^ ]*}} 'p' 'S *'

  // FIXME: there is no mention that this used the template keyword.
  r->template U<int>::~U();
  // CHECK: CXXMemberCallExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:26> 'void'
  // CHECK-NEXT: MemberExpr 0x{{[^ ]*}} <col:3, col:24> '<bound member function type>' ->~U 0x{{[^ ]*}}
  // CHECK-NEXT: NestedNameSpecifier TypeSpecWithTemplate 'template U<int>':'U<int>'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'U<int> *' lvalue ParmVar 0x{{[^ ]*}} 'r' 'U<int> *'

  typeid(a);
  // CHECK: CXXTypeidExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:11> 'const std::type_info' lvalue
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:10> 'S' lvalue ParmVar 0x{{[^ ]*}} 'a' 'S'

  // FIXME: no type information is printed for the argument.
  typeid(S);
  // CHECK: CXXTypeidExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:11> 'const std::type_info' lvalue
}

template <typename... Ts>
void PrimaryExpressions(Ts... a) {
  struct V {
    void f() {
      this;
      // CHECK: CXXThisExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:7> 'V *' this
      [this]{};
      // CHECK: LambdaExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:7, col:14>
      // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:7> col:7 implicit class definition
      // CHECK-NEXT: DefinitionData lambda
      // CHECK-NEXT: DefaultConstructor
      // CHECK-NEXT: CopyConstructor
      // CHECK-NEXT: MoveConstructor
      // CHECK-NEXT: CopyAssignment
      // CHECK-NEXT: MoveAssignment
      // CHECK-NEXT: Destructor
      // CHECK-NEXT: CXXMethodDecl
      // CHECK-NEXT: CompoundStmt
      // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <col:8> col:8 implicit 'V *'
      // CHECK-NEXT: ParenListExpr
      // CHECK-NEXT: CXXThisExpr 0x{{[^ ]*}} <col:8> 'V *' this

      [*this]{};
      // CHECK: LambdaExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:7, col:15>
      // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:7> col:7 implicit class definition
      // CHECK-NEXT: DefinitionData lambda
      // CHECK-NEXT: DefaultConstructor
      // CHECK-NEXT: CopyConstructor
      // CHECK-NEXT: MoveConstructor
      // CHECK-NEXT: CopyAssignment
      // CHECK-NEXT: MoveAssignment
      // CHECK-NEXT: Destructor
      // CHECK-NEXT: CXXMethodDecl
      // CHECK-NEXT: CompoundStmt
      // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <col:8> col:8 implicit 'V'
      // CHECK-NEXT: ParenListExpr 0x{{[^ ]*}} <col:8> 'NULL TYPE'
      // CHECK-NEXT: UnaryOperator 0x{{[^ ]*}} <col:8> 'V' lvalue prefix '*' cannot overflow
      // CHECK-NEXT: CXXThisExpr 0x{{[^ ]*}} <col:8> 'V *' this
    }
  };

  int b, c;

  [](){};
  // CHECK: LambdaExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:8> '(lambda at {{.*}}:[[@LINE-1]]:3)'
  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:3> col:3 implicit class definition
  // CHECK-NEXT: DefinitionData lambda
  // CHECK-NEXT: DefaultConstructor
  // CHECK-NEXT: CopyConstructor
  // CHECK-NEXT: MoveConstructor
  // CHECK-NEXT: CopyAssignment
  // CHECK-NEXT: MoveAssignment
  // CHECK-NEXT: Destructor
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:6, col:8> col:3 operator() 'auto () const' inline
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: CXXConversionDecl 0x{{[^ ]*}} <col:3, col:8> col:3 implicit constexpr operator auto (*)() 'auto (*() const noexcept)()' inline
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:3, col:8> col:3 implicit __invoke 'auto ()' static inline
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:7, col:8>

  [](int a, ...){};
  // CHECK: LambdaExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:18> '(lambda at {{.*}}:[[@LINE-1]]:3)'
  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:3> col:3 implicit class definition
  // CHECK-NEXT: DefinitionData lambda
  // CHECK-NEXT: DefaultConstructor
  // CHECK-NEXT: CopyConstructor
  // CHECK-NEXT: MoveConstructor
  // CHECK-NEXT: CopyAssignment
  // CHECK-NEXT: MoveAssignment
  // CHECK-NEXT: Destructor
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:16, col:18> col:3 operator() 'auto (int, ...) const' inline
  // CHECK-NEXT: ParmVarDecl 0x{{[^ ]*}} <col:6, col:10> col:10 a 'int'
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: CXXConversionDecl 0x{{[^ ]*}} <col:3, col:18> col:3 implicit constexpr operator auto (*)(int, ...) 'auto (*() const noexcept)(int, ...)' inline
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:3, col:18> col:3 implicit __invoke 'auto (int, ...)' static inline
  // CHECK-NEXT: ParmVarDecl 0x{{[^ ]*}} <col:6, col:10> col:10 a 'int'
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:17, col:18>

  [a...]{};
  // CHECK: LambdaExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:10> '(lambda at {{.*}}:[[@LINE-1]]:3)'
  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:3> col:3 implicit class definition
  // CHECK-NEXT: DefinitionData lambda
  // CHECK-NEXT: DefaultConstructor
  // CHECK-NEXT: CopyConstructor
  // CHECK-NEXT: MoveConstructor
  // CHECK-NEXT: CopyAssignment
  // CHECK-NEXT: MoveAssignment
  // CHECK-NEXT: Destructor
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:8, col:10> col:3 operator() 'auto () const -> auto' inline
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <col:4> col:4 implicit 'Ts...'
  // CHECK-NEXT: ParenListExpr 0x{{[^ ]*}} <col:4> 'NULL TYPE'
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:4> 'Ts' lvalue ParmVar 0x{{[^ ]*}} 'a' 'Ts...'
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:9, col:10>

  [=]{};
  // CHECK: LambdaExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> '(lambda at {{.*}}:[[@LINE-1]]:3)'
  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:3> col:3 implicit class definition
  // CHECK-NEXT: DefinitionData lambda
  // CHECK-NEXT: DefaultConstructor
  // CHECK-NEXT: CopyConstructor
  // CHECK-NEXT: MoveConstructor
  // CHECK-NEXT: CopyAssignment
  // CHECK-NEXT: MoveAssignment
  // CHECK-NEXT: Destructor
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:5, col:7> col:3 operator() 'auto () const -> auto' inline
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:6, col:7>

  [=] { return b; };
  // CHECK: LambdaExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:19> '(lambda at {{.*}}:[[@LINE-1]]:3)'
  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:3> col:3 implicit class definition
  // CHECK-NEXT: DefinitionData lambda
  // CHECK-NEXT: DefaultConstructor
  // CHECK-NEXT: CopyConstructor
  // CHECK-NEXT: MoveConstructor
  // CHECK-NEXT: CopyAssignment
  // CHECK-NEXT: MoveAssignment
  // CHECK-NEXT: Destructor
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:5, col:19> col:3 operator() 'auto () const -> auto' inline
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: ReturnStmt 0x{{[^ ]*}} <col:9, col:16>
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:16> 'const int' lvalue Var 0x{{[^ ]*}} 'b' 'int'
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:7, col:19>
  // CHECK-NEXT: ReturnStmt 0x{{[^ ]*}} <col:9, col:16>
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:16> 'const int' lvalue Var 0x{{[^ ]*}} 'b' 'int'

  [&]{};
  // CHECK: LambdaExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> '(lambda at {{.*}}:[[@LINE-1]]:3)'
  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:3> col:3 implicit class definition
  // CHECK-NEXT: DefinitionData lambda
  // CHECK-NEXT: DefaultConstructor
  // CHECK-NEXT: CopyConstructor
  // CHECK-NEXT: MoveConstructor
  // CHECK-NEXT: CopyAssignment
  // CHECK-NEXT: MoveAssignment
  // CHECK-NEXT: Destructor
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:5, col:7> col:3 operator() 'auto () const -> auto' inline
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:6, col:7>

  [&] { return c; };
  // CHECK: LambdaExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:19> '(lambda at {{.*}}:[[@LINE-1]]:3)'
  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:3> col:3 implicit class definition
  // CHECK-NEXT: DefinitionData lambda
  // CHECK-NEXT: DefaultConstructor
  // CHECK-NEXT: CopyConstructor
  // CHECK-NEXT: MoveConstructor
  // CHECK-NEXT: CopyAssignment
  // CHECK-NEXT: MoveAssignment
  // CHECK-NEXT: Destructor
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:5, col:19> col:3 operator() 'auto () const -> auto' inline
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: ReturnStmt 0x{{[^ ]*}} <col:9, col:16>
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:16> 'int' lvalue Var 0x{{[^ ]*}} 'c' 'int'
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:7, col:19>
  // CHECK-NEXT: ReturnStmt 0x{{[^ ]*}} <col:9, col:16>
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:16> 'int' lvalue Var 0x{{[^ ]*}} 'c' 'int'

  [b, &c]{ return b + c; };
  // CHECK: LambdaExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:26> '(lambda at {{.*}}:[[@LINE-1]]:3)'
  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:3> col:3 implicit class definition
  // CHECK-NEXT: DefinitionData lambda
  // CHECK-NEXT: DefaultConstructor
  // CHECK-NEXT: CopyConstructor
  // CHECK-NEXT: MoveConstructor
  // CHECK-NEXT: CopyAssignment
  // CHECK-NEXT: MoveAssignment
  // CHECK-NEXT: Destructor
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:9, col:26> col:3 operator() 'auto () const -> auto' inline
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: ReturnStmt 0x{{[^ ]*}} <col:12, col:23>
  // CHECK-NEXT: BinaryOperator 0x{{[^ ]*}} <col:19, col:23> 'int' '+'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:19> 'const int' lvalue Var 0x{{[^ ]*}} 'b' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:23> 'int' lvalue Var 0x{{[^ ]*}} 'c' 'int'
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <col:4> col:4 implicit 'int'
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <col:8> col:8 implicit 'int &'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:4> 'int' lvalue Var 0x{{[^ ]*}} 'b' 'int'
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:8> 'int' lvalue Var 0x{{[^ ]*}} 'c' 'int'
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:10, col:26>
  // CHECK-NEXT: ReturnStmt 0x{{[^ ]*}} <col:12, col:23>
  // CHECK-NEXT: BinaryOperator 0x{{[^ ]*}} <col:19, col:23> 'int' '+'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:19> 'const int' lvalue Var 0x{{[^ ]*}} 'b' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:23> 'int' lvalue Var 0x{{[^ ]*}} 'c' 'int'

  [a..., x = 12]{};
  // CHECK: LambdaExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:18> '(lambda at {{.*}}:[[@LINE-1]]:3)'
  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:3> col:3 implicit class definition
  // CHECK-NEXT: DefinitionData lambda
  // CHECK-NEXT: DefaultConstructor
  // CHECK-NEXT: CopyConstructor
  // CHECK-NEXT: MoveConstructor
  // CHECK-NEXT: CopyAssignment
  // CHECK-NEXT: MoveAssignment
  // CHECK-NEXT: Destructor
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:16, col:18> col:3 operator() 'auto () const -> auto' inline
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <col:4> col:4 implicit 'Ts...'
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <col:10> col:10 implicit 'int'
  // CHECK-NEXT: ParenListExpr 0x{{[^ ]*}} <col:4> 'NULL TYPE'
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:4> 'Ts' lvalue ParmVar 0x{{[^ ]*}} 'a' 'Ts...'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:14> 'int' 12
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:17, col:18>

  []() constexpr {};
  // CHECK: LambdaExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:19> '(lambda at {{.*}}:[[@LINE-1]]:3)'
  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:3> col:3 implicit class definition
  // CHECK-NEXT: DefinitionData lambda
  // CHECK-NEXT: DefaultConstructor
  // CHECK-NEXT: CopyConstructor
  // CHECK-NEXT: MoveConstructor
  // CHECK-NEXT: CopyAssignment
  // CHECK-NEXT: MoveAssignment
  // CHECK-NEXT: Destructor
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:8, col:19> col:3 constexpr operator() 'auto () const' inline
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: CXXConversionDecl 0x{{[^ ]*}} <col:3, col:19> col:3 implicit constexpr operator auto (*)() 'auto (*() const noexcept)()' inline
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:3, col:19> col:3 implicit constexpr __invoke 'auto ()' static inline
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:18, col:19>

  []() mutable {};
  // CHECK: LambdaExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:17> '(lambda at {{.*}}:[[@LINE-1]]:3)'
  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:3> col:3 implicit class definition
  // CHECK-NEXT: DefinitionData lambda
  // CHECK-NEXT: DefaultConstructor
  // CHECK-NEXT: CopyConstructor
  // CHECK-NEXT: MoveConstructor
  // CHECK-NEXT: CopyAssignment
  // CHECK-NEXT: MoveAssignment
  // CHECK-NEXT: Destructor
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:8, col:17> col:3 operator() 'auto ()' inline
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: CXXConversionDecl 0x{{[^ ]*}} <col:3, col:17> col:3 implicit constexpr operator auto (*)() 'auto (*() const noexcept)()' inline
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:3, col:17> col:3 implicit __invoke 'auto ()' static inline
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:16, col:17>

  []() noexcept {};
  // CHECK: LambdaExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:18> '(lambda at {{.*}}:[[@LINE-1]]:3)'
  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:3> col:3 implicit class definition
  // CHECK-NEXT: DefinitionData lambda
  // CHECK-NEXT: DefaultConstructor
  // CHECK-NEXT: CopyConstructor
  // CHECK-NEXT: MoveConstructor
  // CHECK-NEXT: CopyAssignment
  // CHECK-NEXT: MoveAssignment
  // CHECK-NEXT: Destructor
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:8, col:18> col:3 operator() 'auto () const noexcept' inline
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: CXXConversionDecl 0x{{[^ ]*}} <col:3, col:18> col:3 implicit constexpr operator auto (*)() noexcept 'auto (*() const noexcept)() noexcept' inline
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:3, col:18> col:3 implicit __invoke 'auto () noexcept' static inline
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:17, col:18>

  []() -> int { return 0; };
  // CHECK: LambdaExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:27> '(lambda at {{.*}}:[[@LINE-1]]:3)'
  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:3> col:3 implicit class definition
  // CHECK-NEXT: DefinitionData lambda
  // CHECK-NEXT: DefaultConstructor
  // CHECK-NEXT: CopyConstructor
  // CHECK-NEXT: MoveConstructor
  // CHECK-NEXT: CopyAssignment
  // CHECK-NEXT: MoveAssignment
  // CHECK-NEXT: Destructor
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:11, col:27> col:3 operator() 'auto () const -> int' inline
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: ReturnStmt 0x{{[^ ]*}} <col:17, col:24>
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:24> 'int' 0
  // CHECK-NEXT: CXXConversionDecl 0x{{[^ ]*}} <col:3, col:27> col:3 implicit constexpr operator int (*)() 'auto (*() const noexcept)() -> int' inline
  // CHECK-NEXT: CXXMethodDecl 0x{{[^ ]*}} <col:3, col:27> col:3 implicit __invoke 'auto () -> int' static inline
  // CHECK-NEXT: CompoundStmt 0x{{[^ ]*}} <col:15, col:27>
  // CHECK-NEXT: ReturnStmt 0x{{[^ ]*}} <col:17, col:24>
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:24> 'int' 0

  (a + ...);
  // CHECK: CXXFoldExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:11> '<dependent type>'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:4> 'Ts' lvalue ParmVar 0x{{[^ ]*}} 'a' 'Ts...'
  // CHECK-NEXT: <<<NULL>>>

  (... + a);
  // CHECK: CXXFoldExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:11> '<dependent type>'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:10> 'Ts' lvalue ParmVar 0x{{[^ ]*}} 'a' 'Ts...'

  (a + ... + b);
  // CHECK: CXXFoldExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:15> '<dependent type>'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:4> 'Ts' lvalue ParmVar 0x{{[^ ]*}} 'a' 'Ts...'
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:14> 'int' lvalue Var 0x{{[^ ]*}} 'b' 'int'
}


namespace NS {
struct X {};
void f(X);
void y(...);
} // namespace NS

// CHECK-LABEL: FunctionDecl 0x{{[^ ]*}} {{.*}}ADLCall 'void ()'
void ADLCall() {
  NS::X x;
  // CHECK: CallExpr 0x{{[^ ]*}} <line:[[@LINE+1]]:{{[^>]+}}> 'void' adl{{$}}
  f(x);
  // CHECK: CallExpr 0x{{[^ ]*}} <line:[[@LINE+1]]:{{[^>]+}}> 'void' adl{{$}}
  y(x);
}

// CHECK-LABEL: FunctionDecl 0x{{[^ ]*}} {{.*}}NonADLCall 'void ()'
void NonADLCall() {
  NS::X x;
  // CHECK: CallExpr 0x{{[^ ]*}} <line:[[@LINE+1]]:{{[^>]+}}> 'void'{{$}}
  NS::f(x);
}

// CHECK-LABEL: FunctionDecl 0x{{[^ ]*}} {{.*}}NonADLCall2 'void ()'
void NonADLCall2() {
  NS::X x;
  using NS::f;
  // CHECK: CallExpr 0x{{[^ ]*}} <line:[[@LINE+1]]:{{[^>]+}}> 'void'{{$}}
  f(x);
  // CHECK: CallExpr 0x{{[^ ]*}} <line:[[@LINE+1]]:{{[^>]+}}> 'void' adl{{$}}
  y(x);
}

namespace test_adl_call_three {
using namespace NS;
// CHECK-LABEL: FunctionDecl 0x{{[^ ]*}} {{.*}}NonADLCall3 'void ()'
void NonADLCall3() {
  X x;
  // CHECK: CallExpr 0x{{[^ ]*}} <line:[[@LINE+1]]:{{[^>]+}}> 'void'{{$}}
  f(x);
}
} // namespace test_adl_call_three

namespace GH35300 {
struct Sock {};
void leakNewFn() { new struct Sock; }
// CHECK: CXXNewExpr {{.*}} <col:20, col:31> 'struct Sock *'
}

