// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s \
// RUN: | FileCheck --strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-unknown -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

void test_func() {
  int a, b, c;
  // CHECK: DeclStmt 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:14>
  // CHECK-NEXT: VarDecl 0x{{[^ ]*}} <col:3, col:7> col:7 a 'int'
  // CHECK-NEXT: VarDecl 0x{{[^ ]*}} <col:3, col:10> col:10 b 'int'
  // CHECK-NEXT: VarDecl 0x{{[^ ]*}} <col:3, col:13> col:13 c 'int'
  void d(), e(int);
  // CHECK: DeclStmt 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:19>
  // CHECK-NEXT: FunctionDecl 0x{{[^ ]*}} parent 0x{{[^ ]*}} <col:3, col:10> col:8 d 'void ()'
  // CHECK-NEXT: FunctionDecl 0x{{[^ ]*}} parent 0x{{[^ ]*}} <col:3, col:18> col:13 e 'void (int)'
  // CHECK-NEXT: ParmVarDecl 0x{{[^ ]*}} <col:15> col:18 'int'
  int f;
  // CHECK: DeclStmt 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:8>
  // CHECK-NEXT: VarDecl 0x{{[^ ]*}} <col:3, col:7> col:7 f 'int'
}

// FIXME: These currently do not show up as a DeclStmt.
int a, b, c;
// CHECK: VarDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:5> col:5 a 'int'
// CHECK-NEXT: VarDecl 0x{{[^ ]*}} <col:1, col:8> col:8 b 'int'
// CHECK-NEXT: VarDecl 0x{{[^ ]*}} <col:1, col:11> col:11 c 'int'
void d(), e(int);
// CHECK: FunctionDecl 0x{{[^ ]*}} prev 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:8> col:6 d 'void ()'
// CHECK-NEXT: FunctionDecl 0x{{[^ ]*}} prev 0x{{[^ ]*}} <col:1, col:16> col:11 e 'void (int)'
// CHECK-NEXT: ParmVarDecl 0x{{[^ ]*}} <col:13> col:16 'int'
int f;
// CHECK: VarDecl 0x{{[^ ]*}} <line:[[@LINE-1]]:1, col:5> col:5 f 'int'

