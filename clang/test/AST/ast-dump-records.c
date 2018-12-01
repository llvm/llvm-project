// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s | FileCheck -strict-whitespace %s

struct A;
// CHECK: RecordDecl 0x{{[^ ]*}} <{{.*}}:1, col:8> col:8 struct A

struct B;
// CHECK: RecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:8> col:8 struct B

struct A {
  // CHECK: RecordDecl 0x{{[^ ]*}} prev 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+20]]:1> line:[[@LINE-1]]:8 struct A definition
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
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:9> col:3 'int'
  // CHECK-NEXT: ConstantExpr 0x{{[^ ]*}} <col:9> 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:9> 'int' 0
  int e : 10;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:11> col:7 e 'int'
  // CHECK-NEXT: ConstantExpr 0x{{[^ ]*}} <col:11> 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:11> 'int' 10
  struct B *f;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:13> col:13 f 'struct B *'
};

struct C {
  // CHECK: RecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+36]]:1> line:[[@LINE-1]]:8 struct C definition
  struct {
    // CHECK-NEXT: RecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, line:[[@LINE+3]]:3> line:[[@LINE-1]]:3 struct definition
    int a;
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:9> col:9 a 'int'
  } b;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-5]]:3, line:[[@LINE-1]]:5> col:5 b 'struct (anonymous struct at {{.*}}:[[@LINE-5]]:3)':'struct C::(anonymous at {{.*}}:[[@LINE-5]]:3)'

  union {
    // CHECK-NEXT: RecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, line:[[@LINE+5]]:3> line:[[@LINE-1]]:3 union definition
    int c;
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:9> col:9 c 'int'
    float d;
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:11> col:11 d 'float'
  };
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-7]]:3> col:3 implicit 'union C::(anonymous at {{.*}}:[[@LINE-7]]:3)'
  // CHECK-NEXT: IndirectFieldDecl 0x{{[^ ]*}} <line:[[@LINE-6]]:9> col:9 implicit c 'int'
  // CHECK-NEXT: Field 0x{{[^ ]*}} '' 'union C::(anonymous at {{.*}}:[[@LINE-9]]:3)'
  // CHECK-NEXT: Field 0x{{[^ ]*}} 'c' 'int'
  // CHECK-NEXT: IndirectFieldDecl 0x{{[^ ]*}} <line:[[@LINE-7]]:11> col:11 implicit d 'float'
  // CHECK-NEXT: Field 0x{{[^ ]*}} '' 'union C::(anonymous at {{.*}}:[[@LINE-12]]:3)'
  // CHECK-NEXT: Field 0x{{[^ ]*}} 'd' 'float'

  struct {
    // CHECK-NEXT: RecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, line:[[@LINE+4]]:3> line:[[@LINE-1]]:3 struct definition
    int e, f;
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:9> col:9 e 'int'
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <col:5, col:12> col:12 f 'int'
  };
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-6]]:3> col:3 implicit 'struct C::(anonymous at {{.*}}:[[@LINE-6]]:3)'
  // CHECK-NEXT: IndirectFieldDecl 0x{{[^ ]*}} <line:[[@LINE-5]]:9> col:9 implicit e 'int'
  // CHECK-NEXT: Field 0x{{[^ ]*}} '' 'struct C::(anonymous at {{.*}}:[[@LINE-8]]:3)'
  // CHECK-NEXT: Field 0x{{[^ ]*}} 'e' 'int'
  // CHECK-NEXT: IndirectFieldDecl 0x{{[^ ]*}} <col:12> col:12 implicit f 'int'
  // CHECK-NEXT: Field 0x{{[^ ]*}} '' 'struct C::(anonymous at {{.*}}:[[@LINE-11]]:3)'
  // CHECK-NEXT: Field 0x{{[^ ]*}} 'f' 'int'
};

struct D {
  // CHECK-NEXT: RecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+7]]:1> line:[[@LINE-1]]:8 struct D definition
  int a;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:7> col:7 a 'int'
  int b[10];
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:11> col:7 b 'int [10]'
  int c[];
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:9> col:7 c 'int []'
};

union E;
// CHECK: RecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:7> col:7 union E

union F;
// CHECK: RecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:7> col:7 union F

union E {
  // CHECK: RecordDecl 0x{{[^ ]*}} prev 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+20]]:1> line:[[@LINE-1]]:7 union E definition
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
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:9> col:3 'int'
  // CHECK-NEXT: ConstantExpr 0x{{[^ ]*}} <col:9> 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:9> 'int' 0
  int e : 10;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:11> col:7 e 'int'
  // CHECK-NEXT: ConstantExpr 0x{{[^ ]*}} <col:11> 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:11> 'int' 10
  struct B *f;
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:13> col:13 f 'struct B *'
};

union G {
  // CHECK: RecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, line:[[@LINE+38]]:1> line:[[@LINE-1]]:7 union G definition
  struct {
    // CHECK-NEXT: RecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, line:[[@LINE+3]]:3> line:[[@LINE-1]]:3 struct definition
    int a;
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:9> col:9 a 'int'
  } b;
  // FIXME: note that it talks about 'struct G' below; the same happens in
  // other cases with union G as well.
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-7]]:3, line:[[@LINE-3]]:5> col:5 b 'struct (anonymous struct at {{.*}}:[[@LINE-7]]:3)':'struct G::(anonymous at {{.*}}:[[@LINE-7]]:3)'

  union {
    // CHECK-NEXT: RecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, line:[[@LINE+5]]:3> line:[[@LINE-1]]:3 union definition
    int c;
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:9> col:9 c 'int'
    float d;
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:11> col:11 d 'float'
  };
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-7]]:3> col:3 implicit 'union G::(anonymous at {{.*}}:[[@LINE-7]]:3)'
  // CHECK-NEXT: IndirectFieldDecl 0x{{[^ ]*}} <line:[[@LINE-6]]:9> col:9 implicit c 'int'
  // CHECK-NEXT: Field 0x{{[^ ]*}} '' 'union G::(anonymous at {{.*}}:[[@LINE-9]]:3)'
  // CHECK-NEXT: Field 0x{{[^ ]*}} 'c' 'int'
  // CHECK-NEXT: IndirectFieldDecl 0x{{[^ ]*}} <line:[[@LINE-7]]:11> col:11 implicit d 'float'
  // CHECK-NEXT: Field 0x{{[^ ]*}} '' 'union G::(anonymous at {{.*}}:[[@LINE-12]]:3)'
  // CHECK-NEXT: Field 0x{{[^ ]*}} 'd' 'float'

  struct {
    // CHECK-NEXT: RecordDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:3, line:[[@LINE+4]]:3> line:[[@LINE-1]]:3 struct definition
    int e, f;
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:5, col:9> col:9 e 'int'
    // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <col:5, col:12> col:12 f 'int'
  };
  // CHECK-NEXT: FieldDecl 0x{{[^ ]*}} <line:[[@LINE-6]]:3> col:3 implicit 'struct G::(anonymous at {{.*}}:[[@LINE-6]]:3)'
  // CHECK-NEXT: IndirectFieldDecl 0x{{[^ ]*}} <line:[[@LINE-5]]:9> col:9 implicit e 'int'
  // CHECK-NEXT: Field 0x{{[^ ]*}} '' 'struct G::(anonymous at {{.*}}:[[@LINE-8]]:3)'
  // CHECK-NEXT: Field 0x{{[^ ]*}} 'e' 'int'
  // CHECK-NEXT: IndirectFieldDecl 0x{{[^ ]*}} <col:12> col:12 implicit f 'int'
  // CHECK-NEXT: Field 0x{{[^ ]*}} '' 'struct G::(anonymous at {{.*}}:[[@LINE-11]]:3)'
  // CHECK-NEXT: Field 0x{{[^ ]*}} 'f' 'int'
};

