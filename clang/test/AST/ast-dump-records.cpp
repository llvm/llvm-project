// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++17 -ast-dump %s | FileCheck -strict-whitespace %s

struct A;
// CHECK: CXXRecordDecl 0x{{[^ ]*}} <{{.*}}:1, col:8> col:8 struct A

struct B;
// CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:8> col:8 referenced struct B

struct A {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} prev 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+29]]:1> line:[[@LINE-1]]:8 struct A definition
  // CHECK-NEXT: DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
  // CHECK-NEXT: DefaultConstructor exists trivial needs_implicit
  // CHECK-NEXT: CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
  // CHECK-NEXT: MoveConstructor exists simple trivial needs_implicit
  // CHECK-NEXT: CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
  // CHECK-NEXT: MoveAssignment exists simple trivial needs_implicit
  // CHECK-NEXT: Destructor simple irrelevant trivial needs_implicit

  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:1, col:8> col:8 implicit struct A
  int a;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> col:7 a 'int'
  int b, c;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> col:7 b 'int'
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <col:3, col:10> col:10 c 'int'
  int d : 12;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:11> col:7 d 'int'
  // CHECK-NEXT: ConstantExpr 0x{{[^ ]*}} <col:11> 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:11> 'int' 12
  int : 0;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:9> col:7 'int'
  // CHECK-NEXT: ConstantExpr 0x{{[^ ]*}} <col:9> 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:9> 'int' 0
  int e : 10;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:11> col:7 e 'int'
  // CHECK-NEXT: ConstantExpr 0x{{[^ ]*}} <col:11> 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:11> 'int' 10
  B *f;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:6> col:6 f 'B *'
};

struct C {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+66]]:1> line:[[@LINE-1]]:8 struct C definition
  // CHECK-NEXT: DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal has_variant_members
  // CHECK-NEXT: DefaultConstructor exists trivial needs_implicit
  // CHECK-NEXT: CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
  // CHECK-NEXT: MoveConstructor exists simple trivial needs_implicit
  // CHECK-NEXT: CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
  // CHECK-NEXT: MoveAssignment exists simple trivial needs_implicit
  // CHECK-NEXT: Destructor simple irrelevant trivial needs_implicit

  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:1, col:8> col:8 implicit struct C
  struct {
    // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, line:[[@LINE+10]]:3> line:[[@LINE-1]]:3 struct definition
    // CHECK-NEXT: DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
    // CHECK-NEXT: DefaultConstructor exists trivial needs_implicit
    // CHECK-NEXT: CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
    // CHECK-NEXT: MoveConstructor exists simple trivial needs_implicit
    // CHECK-NEXT: CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
    // CHECK-NEXT: MoveAssignment exists simple trivial needs_implicit
    // CHECK-NEXT: Destructor simple irrelevant trivial needs_implicit
    int a;
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:9> col:9 a 'int'
  } b;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-12]]:3, line:[[@LINE-1]]:5> col:5 b 'struct (anonymous struct at {{.*}}:[[@LINE-12]]:3)':'C::(anonymous struct at {{.*}}:[[@LINE-12]]:3)'

  union {
    // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, line:[[@LINE+12]]:3> line:[[@LINE-1]]:3 union definition
    // CHECK-NEXT: DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
    // CHECK-NEXT: DefaultConstructor exists trivial needs_implicit
    // CHECK-NEXT: CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
    // CHECK-NEXT: MoveConstructor exists simple trivial needs_implicit
    // CHECK-NEXT: CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
    // CHECK-NEXT: MoveAssignment exists simple trivial needs_implicit
    // CHECK-NEXT: Destructor simple irrelevant trivial needs_implicit
    int c;
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:9> col:9 c 'int'
    float d;
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:11> col:11 d 'float'
  };
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-14]]:3> col:3 implicit 'C::(anonymous union at {{.*}}:[[@LINE-14]]:3)'
  // CHECK-NEXT: IndirectFieldDecl 0x{{[^ ]*}} <line:[[@LINE-6]]:9> col:9 implicit c 'int'
  // CHECK-NEXT: Field 0x{{[^ ]*}} '' 'C::(anonymous union at {{.*}}:[[@LINE-16]]:3)'
  // CHECK-NEXT: Field 0x{{[^ ]*}} 'c' 'int'
  // CHECK-NEXT: IndirectFieldDecl 0x{{[^ ]*}} <line:[[@LINE-7]]:11> col:11 implicit d 'float'
  // CHECK-NEXT: Field 0x{{[^ ]*}} '' 'C::(anonymous union at {{.*}}:[[@LINE-19]]:3)'
  // CHECK-NEXT: Field 0x{{[^ ]*}} 'd' 'float'

  struct {
    // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, line:[[@LINE+11]]:3> line:[[@LINE-1]]:3 struct definition
    // CHECK-NEXT: DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
    // CHECK-NEXT: DefaultConstructor exists trivial needs_implicit
    // CHECK-NEXT: CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
    // CHECK-NEXT: MoveConstructor exists simple trivial needs_implicit
    // CHECK-NEXT: CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
    // CHECK-NEXT: MoveAssignment exists simple trivial needs_implicit
    // CHECK-NEXT: Destructor simple irrelevant trivial needs_implicit
    int e, f;
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:9> col:9 e 'int'
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <col:5, col:12> col:12 f 'int'
  };
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-13]]:3> col:3 implicit 'C::(anonymous struct at {{.*}}:[[@LINE-13]]:3)'
  // CHECK-NEXT: IndirectFieldDecl 0x{{[^ ]*}} <line:[[@LINE-5]]:9> col:9 implicit e 'int'
  // CHECK-NEXT: Field 0x{{[^ ]*}} '' 'C::(anonymous struct at {{.*}}:[[@LINE-15]]:3)'
  // CHECK-NEXT: Field 0x{{[^ ]*}} 'e' 'int'
  // CHECK-NEXT: IndirectFieldDecl 0x{{[^ ]*}} <col:12> col:12 implicit f 'int'
  // CHECK-NEXT: Field 0x{{[^ ]*}} '' 'C::(anonymous struct at {{.*}}:[[@LINE-18]]:3)'
  // CHECK-NEXT: Field 0x{{[^ ]*}} 'f' 'int'
};

struct D {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+16]]:1> line:[[@LINE-1]]:8 struct D definition
  // CHECK-NEXT: DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
  // CHECK-NEXT: DefaultConstructor exists trivial needs_implicit
  // CHECK-NEXT: CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
  // CHECK-NEXT: MoveConstructor exists simple trivial needs_implicit
  // CHECK-NEXT: CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
  // CHECK-NEXT: MoveAssignment exists simple trivial needs_implicit
  // CHECK-NEXT: Destructor simple irrelevant trivial needs_implicit

  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:1, col:8> col:8 implicit struct D
  int a;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> col:7 a 'int'
  int b[10];
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:11> col:7 b 'int [10]'
  int c[];
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:9> col:7 c 'int []'
};

union E;
// CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:7> col:7 union E

union F;
// CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:7> col:7 union F

union E {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} prev 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+29]]:1> line:[[@LINE-1]]:7 union E definition
  // CHECK-NEXT: DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
  // CHECK-NEXT: DefaultConstructor exists trivial needs_implicit
  // CHECK-NEXT: CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
  // CHECK-NEXT: MoveConstructor exists simple trivial needs_implicit
  // CHECK-NEXT: CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
  // CHECK-NEXT: MoveAssignment exists simple trivial needs_implicit
  // CHECK-NEXT: Destructor simple irrelevant trivial needs_implicit

  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:1, col:7> col:7 implicit union E
  int a;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> col:7 a 'int'
  int b, c;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> col:7 b 'int'
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <col:3, col:10> col:10 c 'int'
  int d : 12;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:11> col:7 d 'int'
  // CHECK-NEXT: ConstantExpr 0x{{[^ ]*}} <col:11> 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:11> 'int' 12
  int : 0;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:9> col:7 'int'
  // CHECK-NEXT: ConstantExpr 0x{{[^ ]*}} <col:9> 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:9> 'int' 0
  int e : 10;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:11> col:7 e 'int'
  // CHECK-NEXT: ConstantExpr 0x{{[^ ]*}} <col:11> 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:11> 'int' 10
  B *f;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:6> col:6 f 'B *'
};

union G {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+71]]:1> line:[[@LINE-1]]:7 union G definition
  // CHECK-NEXT: DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
  // CHECK-NEXT: DefaultConstructor exists trivial needs_implicit
  // CHECK-NEXT: CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
  // CHECK-NEXT: MoveConstructor exists simple trivial needs_implicit
  // CHECK-NEXT: CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
  // CHECK-NEXT: MoveAssignment exists simple trivial needs_implicit
  // CHECK-NEXT: Destructor simple irrelevant trivial needs_implicit

  // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <col:1, col:7> col:7 implicit union G
  struct {
    // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, line:[[@LINE+11]]:3> line:[[@LINE-1]]:3 struct definition
    // CHECK-NEXT: DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
    // CHECK-NEXT: DefaultConstructor exists trivial needs_implicit
    // CHECK-NEXT: CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
    // CHECK-NEXT: MoveConstructor exists simple trivial needs_implicit
    // CHECK-NEXT: CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
    // CHECK-NEXT: MoveAssignment exists simple trivial needs_implicit
    // CHECK-NEXT: Destructor simple irrelevant trivial needs_implicit

    int a;
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:9> col:9 a 'int'
  } b;
  // FIXME: note that it talks about 'struct G' below; the same happens in
  // other cases with union G as well.
  // CHECK: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-15]]:3, line:[[@LINE-3]]:5> col:5 b 'struct (anonymous struct at {{.*}}:[[@LINE-15]]:3)':'G::(anonymous struct at {{.*}}:[[@LINE-15]]:3)'

  union {
    // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, line:[[@LINE+13]]:3> line:[[@LINE-1]]:3 union definition
    // CHECK-NEXT: DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
    // CHECK-NEXT: DefaultConstructor exists trivial needs_implicit
    // CHECK-NEXT: CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
    // CHECK-NEXT: MoveConstructor exists simple trivial needs_implicit
    // CHECK-NEXT: CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
    // CHECK-NEXT: MoveAssignment exists simple trivial needs_implicit
    // CHECK-NEXT: Destructor simple irrelevant trivial needs_implicit

    int c;
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:9> col:9 c 'int'
    float d;
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:11> col:11 d 'float'
  };
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-15]]:3> col:3 implicit 'G::(anonymous union at {{.*}}:[[@LINE-15]]:3)'
  // CHECK-NEXT: IndirectFieldDecl 0x{{[^ ]*}} <line:[[@LINE-6]]:9> col:9 implicit c 'int'
  // CHECK-NEXT: Field 0x{{[^ ]*}} '' 'G::(anonymous union at {{.*}}:[[@LINE-17]]:3)'
  // CHECK-NEXT: Field 0x{{[^ ]*}} 'c' 'int'
  // CHECK-NEXT: IndirectFieldDecl 0x{{[^ ]*}} <line:[[@LINE-7]]:11> col:11 implicit d 'float'
  // CHECK-NEXT: Field 0x{{[^ ]*}} '' 'G::(anonymous union at {{.*}}:[[@LINE-20]]:3)'
  // CHECK-NEXT: Field 0x{{[^ ]*}} 'd' 'float'

  struct {
    // CHECK-NEXT: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, line:[[@LINE+12]]:3> line:[[@LINE-1]]:3 struct definition
    // CHECK-NEXT: DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
    // CHECK-NEXT: DefaultConstructor exists trivial needs_implicit
    // CHECK-NEXT: CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
    // CHECK-NEXT: MoveConstructor exists simple trivial needs_implicit
    // CHECK-NEXT: CopyAssignment trivial has_const_param needs_implicit implicit_has_const_param
    // CHECK-NEXT: MoveAssignment exists simple trivial needs_implicit
    // CHECK-NEXT: Destructor simple irrelevant trivial needs_implicit

    int e, f;
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:9> col:9 e 'int'
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <col:5, col:12> col:12 f 'int'
  };
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-14]]:3> col:3 implicit 'G::(anonymous struct at {{.*}}:[[@LINE-14]]:3)'
  // CHECK-NEXT: IndirectFieldDecl 0x{{[^ ]*}} <line:[[@LINE-5]]:9> col:9 implicit e 'int'
  // CHECK-NEXT: Field 0x{{[^ ]*}} '' 'G::(anonymous struct at {{.*}}:[[@LINE-16]]:3)'
  // CHECK-NEXT: Field 0x{{[^ ]*}} 'e' 'int'
  // CHECK-NEXT: IndirectFieldDecl 0x{{[^ ]*}} <col:12> col:12 implicit f 'int'
  // CHECK-NEXT: Field 0x{{[^ ]*}} '' 'G::(anonymous struct at {{.*}}:[[@LINE-19]]:3)'
  // CHECK-NEXT: Field 0x{{[^ ]*}} 'f' 'int'
};

struct Base1 {};
struct Base2 {};
struct Base3 {};

struct Derived1 : Base1 {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct Derived1 definition
  // CHECK: public 'Base1'
};

struct Derived2 : private Base1 {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct Derived2 definition
  // CHECK: private 'Base1'
};

struct Derived3 : virtual Base1 {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct Derived3 definition
  // CHECK: virtual public 'Base1'
};

struct Derived4 : Base1, virtual Base2, protected Base3 {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+4]]:1> line:[[@LINE-1]]:8 struct Derived4 definition
  // CHECK: public 'Base1'
  // CHECK-NEXT: virtual public 'Base2'
  // CHECK-NEXT: protected 'Base3'
};

struct Derived5 : protected virtual Base1 {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct Derived5 definition
  // CHECK: virtual protected 'Base1'
};

template <typename... Bases>
struct Derived6 : virtual public Bases... {
  // CHECK: CXXRecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+2]]:1> line:[[@LINE-1]]:8 struct Derived6 definition
  // CHECK: virtual public 'Bases'...
};
