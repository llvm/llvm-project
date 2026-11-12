union S {
  unsigned ac : 4;
  unsigned : 4;
  unsigned clock : 1;
  unsigned : 0;
  unsigned flag : 1;
};

struct X {
  unsigned light : 1;
  unsigned toaster : 1;
  int count;
  union S stat;
};

// RUN: c-index-test -test-print-bitwidth %s | FileCheck %s
// CHECK: FieldDecl=ac:2:12 (Definition) bitwidth=4
// CHECK: FieldDecl=:3:3 (Definition) bitwidth=4
// CHECK: FieldDecl=clock:4:12 (Definition) bitwidth=1
// CHECK: FieldDecl=:5:3 (Definition) bitwidth=0
// CHECK: FieldDecl=flag:6:12 (Definition) bitwidth=1
// CHECK: FieldDecl=light:10:12 (Definition) bitwidth=1
// CHECK: FieldDecl=toaster:11:12 (Definition) bitwidth=1
// CHECK-NOT: count
// CHECK-NOT: stat
