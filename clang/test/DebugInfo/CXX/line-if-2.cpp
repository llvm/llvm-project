// RUN: %clang_cc1 -debug-info-kind=limited -gno-column-info -triple=x86_64-pc-linux -emit-llvm %s -o - | FileCheck  %s

// The important thing is that the compare and the conditional branch have
// locs with the same scope (the lexical block for the 'if'). By turning off
// column info, they end up with the same !dbg record, which halves the number
// of checks to verify the scope.

int c = 2;

int f() {
#line 100
  if (int a = 5; a > c)
    return 1;
  return 0;
}
// CHECK-LABEL: define {{.*}} @_Z1fv()
// CHECK:       = icmp {{.*}} !dbg [[F_CMP:![0-9]+]]
// CHECK-NEXT:  br i1 {{.*}} !dbg [[F_CMP]]

int g() {
#line 200
  if (int a = f())
    return 2;
  return 3;
}
// CHECK-LABEL: define {{.*}} @_Z1gv()
// CHECK:       = icmp {{.*}} !dbg [[G_CMP:![0-9]+]]
// CHECK-NEXT:  br i1 {{.*}} !dbg [[G_CMP]]

int h() {
#line 300
  if (c > 3)
    return 4;
  return 5;
}
// CHECK-LABEL: define {{.*}} @_Z1hv()
// CHECK:       = icmp {{.*}} !dbg [[H_CMP:![0-9]+]]
// CHECK-NEXT:  br i1 {{.*}} !dbg [[H_CMP]]

// CHECK-DAG: [[F_CMP]] = !DILocation(line: 100, scope: [[F_SCOPE:![0-9]+]]
// CHECK-DAG: [[F_SCOPE]] = distinct !DILexicalBlock({{.*}} line: 100)
// CHECK-DAG: [[G_CMP]] = !DILocation(line: 200, scope: [[G_SCOPE:![0-9]+]]
// CHECK-DAG: [[G_SCOPE]] = distinct !DILexicalBlock({{.*}} line: 200)
// CHECK-DAG: [[H_CMP]] = !DILocation(line: 300, scope: [[H_SCOPE:![0-9]+]]
// CHECK-DAG: [[H_SCOPE]] = distinct !DILexicalBlock({{.*}} line: 300)
