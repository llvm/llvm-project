// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -disable-llvm-passes -o - -hlsl-entry main %s | FileCheck %s

template<typename K, typename V>
struct Pair {
  K First;
  V Second;

  K getFirst() {
	  return this.First;
  }

  V getSecond() {
    return Second;
  }
};

[numthreads(1, 1, 1)]
void main() {
  Pair<int, float> Vals = {1, 2.0};
  Vals.First = Vals.getFirst();
  Vals.Second = Vals.getSecond();
}

// CHECK:     -CXXMethodDecl 0x{{[0-9A-Fa-f]+}} <line:8:3, line:10:3> line:8:5 getFirst 'K ()' implicit-inline
// CHECK-NEXT:-CompoundStmt 0x{{[0-9A-Fa-f]+}} <col:16, line:10:3>
// CHECK-NEXT:-ReturnStmt 0x{{[0-9A-Fa-f]+}} <line:9:4, col:16>
// CHECK-NEXT:-CXXDependentScopeMemberExpr 0x{{[0-9A-Fa-f]+}} <col:11, col:16> '<dependent type>' lvalue .First
// CHECK-NEXT:-CXXThisExpr 0x{{[0-9A-Fa-f]+}} <col:11> 'Pair<K, V>' lvalue this
// CHECK-NEXT:-CXXMethodDecl 0x{{[0-9A-Fa-f]+}} <line:12:3, line:14:3> line:12:5 getSecond 'V ()' implicit-inline
// CHECK-NEXT:-CompoundStmt 0x{{[0-9A-Fa-f]+}} <col:17, line:14:3>
// CHECK-NEXT:-ReturnStmt 0x{{[0-9A-Fa-f]+}} <line:13:5, col:12>
// CHECK-NEXT:-MemberExpr 0x{{[0-9A-Fa-f]+}} <col:12> 'V' lvalue .Second 0x{{[0-9A-Fa-f]+}}
// CHECK-NEXT:-CXXThisExpr 0x{{[0-9A-Fa-f]+}} <col:12> 'Pair<K, V>' lvalue implicit this

// CHECK:     -CXXMethodDecl 0x{{[0-9A-Fa-f]+}} <line:8:3, line:10:3> line:8:5 used getFirst 'int ()' implicit_instantiation implicit-inline
// CHECK-NEXT:-CompoundStmt 0x{{[0-9A-Fa-f]+}} <col:16, line:10:3>
// CHECK-NEXT:-ReturnStmt 0x{{[0-9A-Fa-f]+}} <line:9:4, col:16>
// CHECK-NEXT:-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <col:11, col:16> 'int':'int' <LValueToRValue>
// CHECK-NEXT:-MemberExpr 0x{{[0-9A-Fa-f]+}} <col:11, col:16> 'int':'int' lvalue .First 0x{{[0-9A-Fa-f]+}}
// CHECK-NEXT:-CXXThisExpr 0x{{[0-9A-Fa-f]+}} <col:11> 'Pair<int, float>' lvalue this
// CHECK-NEXT:-CXXMethodDecl 0x{{[0-9A-Fa-f]+}} <line:12:3, line:14:3> line:12:5 used getSecond 'float ()' implicit_instantiation implicit-inline
// CHECK-NEXT:-CompoundStmt 0x{{[0-9A-Fa-f]+}} <col:17, line:14:3>
// CHECK-NEXT:-ReturnStmt 0x{{[0-9A-Fa-f]+}} <line:13:5, col:12>
// CHECK-NEXT:-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <col:12> 'float':'float' <LValueToRValue>
// CHECK-NEXT:-MemberExpr 0x{{[0-9A-Fa-f]+}} <col:12> 'float':'float' lvalue .Second 0x{{[0-9A-Fa-f]+}}
// CHECK-NEXT:-CXXThisExpr 0x{{[0-9A-Fa-f]+}} <col:12> 'Pair<int, float>' lvalue implicit this
