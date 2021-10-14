// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu11 -ast-dump %s \
// RUN: | FileCheck --strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu11 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu11 \
// RUN: -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

void Comma(void) {
  1, 2, 3;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:9> 'int' ','
  // CHECK-NEXT: BinaryOperator 0x{{[^ ]*}} <col:3, col:6> 'int' ','
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:3> 'int' 1
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:6> 'int' 2
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:9> 'int' 3
}

void Assignment(int a) {
  a = 12;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> 'int' '='
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:7> 'int' 12

  a += a;
  // CHECK: CompoundAssignOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:8> 'int' '+=' ComputeLHSTy='int' ComputeResultTy='int'
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:8> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
}

void Conditionals(int a) {
  a ? 0 : 1;
  // CHECK: ConditionalOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:11> 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:7> 'int' 0
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:11> 'int' 1

  a ?: 0;
  // CHECK: BinaryConditionalOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:8> 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: OpaqueValueExpr 0x{{[^ ]*}} <col:3> 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: OpaqueValueExpr 0x{{[^ ]*}} <col:3> 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:8> 'int' 0
}

void BinaryOperators(int a, int b) {
  // Logical operators
  a || b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:8> 'int' '||'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:8> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  a && b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:8> 'int' '&&'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:8> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  // Bitwise operators
  a | b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> 'int' '|'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:7> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  a ^ b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> 'int' '^'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:7> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  a & b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> 'int' '&'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:7> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  // Equality operators
  a == b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:8> 'int' '=='
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:8> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  a != b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:8> 'int' '!='
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:8> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  // Relational operators
  a < b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> 'int' '<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:7> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  a > b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> 'int' '>'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:7> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  a <= b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:8> 'int' '<='
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:8> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  a >= b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:8> 'int' '>='
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:8> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  // Bit shifting operators
  a << b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:8> 'int' '<<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:8> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  a >> b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:8> 'int' '>>'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:8> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  // Additive operators
  a + b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> 'int' '+'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:7> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  a - b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> 'int' '-'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:7> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  // Multiplicative operators
  a * b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> 'int' '*'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:7> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  a / b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> 'int' '/'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:7> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'

  a % b;
  // CHECK: BinaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> 'int' '%'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:7> 'int' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int'
}

void UnaryOperators(int a, int *b) {
  // Cast operators
  (float)a;
  // CHECK: CStyleCastExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:10> 'float' <IntegralToFloating>
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:10> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'

  // ++, --, and ~ are covered elsewhere.

  -a;
  // CHECK: UnaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:4> 'int' prefix '-'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:4> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'

  +a;
  // CHECK: UnaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:4> 'int' prefix '+' cannot overflow
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:4> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'

  &a;
  // CHECK: UnaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:4> 'int *' prefix '&' cannot overflow
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:4> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'

  *b;
  // CHECK: ImplicitCastExpr
  // CHECK-NEXT: UnaryOperator 0x{{[^ ]*}} <col:3, col:4> 'int' lvalue prefix '*' cannot overflow
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:4> 'int *' lvalue ParmVar 0x{{[^ ]*}} 'b' 'int *'

  !a;
  // CHECK: UnaryOperator 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:4> 'int' prefix '!' cannot overflow
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:4> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'

  sizeof a;
  // CHECK: UnaryExprOrTypeTraitExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:10> 'unsigned long' sizeof
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:10> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'

  sizeof(int);
  // CHECK: UnaryExprOrTypeTraitExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:13> 'unsigned long' sizeof 'int'

  _Alignof(int);
  // FIXME: Uses C++ spelling for alignof in C mode.
  // CHECK: UnaryExprOrTypeTraitExpr 0x{{[^ ]*}} <line:[[@LINE-2]]:3, col:15> 'unsigned long' alignof 'int'
}

struct S {
  int a;
};

void PostfixOperators(int *a, struct S b, struct S *c) {
  a[0];
  // CHECK: ImplicitCastExpr
  // CHECK-NEXT: ArraySubscriptExpr 0x{{[^ ]*}} <col:3, col:6> 'int' lvalue
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int *' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int *'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:5> 'int' 0

  UnaryOperators(*a, a);
  // CHECK: CallExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:23> 'void'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'void (int, int *)' Function 0x{{[^ ]*}} 'UnaryOperators' 'void (int, int *)'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: UnaryOperator 0x{{[^ ]*}} <col:18, col:19> 'int' lvalue prefix '*' cannot overflow
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:19> 'int *' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int *'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:22> 'int *' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int *'

  b.a;
  // CHECK: ImplicitCastExpr
  // CHECK-NEXT: MemberExpr 0x{{[^ ]*}} <col:3, col:5> 'int' lvalue .a 0x{{[^ ]*}}
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'struct S':'struct S' lvalue ParmVar 0x{{[^ ]*}} 'b' 'struct S':'struct S'

  c->a;
  // CHECK: ImplicitCastExpr
  // CHECK-NEXT: MemberExpr 0x{{[^ ]*}} <col:3, col:6> 'int' lvalue ->a 0x{{[^ ]*}}
  // CHECK: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'struct S *' lvalue ParmVar 0x{{[^ ]*}} 'c' 'struct S *'

  // Postfix ++ and -- are covered elsewhere.

  (int [4]){1, 2, 3, 4, };
  // CHECK: ImplicitCastExpr
  // CHECK-NEXT: CompoundLiteralExpr 0x{{[^ ]*}} <col:3, col:25> 'int [4]' lvalue
  // CHECK-NEXT: InitListExpr 0x{{[^ ]*}} <col:12, col:25> 'int [4]'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:13> 'int' 1
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:16> 'int' 2
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:19> 'int' 3
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:22> 'int' 4

  (struct S){1};
  // CHECK: ImplicitCastExpr
  // CHECK-NEXT: CompoundLiteralExpr 0x{{[^ ]*}} <col:3, col:15> 'struct S':'struct S' lvalue
  // CHECK-NEXT: InitListExpr 0x{{[^ ]*}} <col:13, col:15> 'struct S':'struct S'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:14> 'int' 1
}

enum E { One };

void PrimaryExpressions(int a) {
  a;
  // CHECK: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:3> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'

  'a';
  // CHECK: CharacterLiteral 0x{{[^ ]*}} <line:[[@LINE-1]]:3> 'int' 97

  L'a';
  // CHECK: CharacterLiteral 0x{{[^ ]*}} <line:[[@LINE-1]]:3> 'int' 97

  "a";
  // ImplicitCastExpr
  // CHECK: StringLiteral 0x{{[^ ]*}} <col:3> 'char [2]' lvalue "a"

  L"a";
  // ImplicitCastExpr
  // CHECK: StringLiteral 0x{{[^ ]*}} <col:3> 'int [2]' lvalue L"a"

  u8"a";
  // ImplicitCastExpr
  // CHECK: StringLiteral 0x{{[^ ]*}} <col:3> 'char [2]' lvalue u8"a"

  U"a";
  // ImplicitCastExpr
  // CHECK: StringLiteral 0x{{[^ ]*}} <col:3> 'unsigned int [2]' lvalue U"a"

  u"a";
  // ImplicitCastExpr
  // CHECK: StringLiteral 0x{{[^ ]*}} <col:3> 'unsigned short [2]' lvalue u"a"

  1;
  // CHECK: IntegerLiteral 0x{{[^ ]*}} <line:[[@LINE-1]]:3> 'int' 1

  1u;
  // CHECK: IntegerLiteral 0x{{[^ ]*}} <line:[[@LINE-1]]:3> 'unsigned int' 1

  1ll;
  // CHECK: IntegerLiteral 0x{{[^ ]*}} <line:[[@LINE-1]]:3> 'long long' 1

  1.0;
  // CHECK: FloatingLiteral 0x{{[^ ]*}} <line:[[@LINE-1]]:3> 'double' {{1\.[0]*e[\+]?[0]+}}

  1.0f;
  // CHECK: FloatingLiteral 0x{{[^ ]*}} <line:[[@LINE-1]]:3> 'float' {{1\.[0]*e[\+]?[0]+}}

  1.0l;
  // CHECK: FloatingLiteral 0x{{[^ ]*}} <line:[[@LINE-1]]:3> 'long double' {{1\.[0]*e[\+]?[0]+}}

  One;
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3> 'int' EnumConstant 0x{{[^ ]*}} 'One' 'int'

  (a);
  // CHECK: ImplicitCastExpr
  // CHECK-NEXT: ParenExpr 0x{{[^ ]*}} <col:3, col:5> 'int' lvalue
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:4> 'int' lvalue ParmVar 0x{{[^ ]*}} 'a' 'int'

  // Generic selection expressions are covered elsewhere.
}
