// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -disable-llvm-passes -o - -hlsl-entry main %s | FileCheck %s

class Pair {
  int First;
  int Second;

  int getFirst() {
	  return this.First;
  }

  int getSecond() {
    return Second;
  }
};

class PairInfo : Pair {
  int Sum;

  int getSum() {
    return this.First + Second;
  }
};

[numthreads(1, 1, 1)]
void main() {
  Pair Vals = {1, 2};
  Vals.First = Vals.getFirst();
  Vals.Second = Vals.getSecond();

  PairInfo ValsInfo;
  ValsInfo.First = Vals.First;
  ValsInfo.Second = Vals.Second;
  ValsInfo.Sum = ValsInfo.getSum();

}

// CHECK:     -CXXMethodDecl 0x{{[0-9A-Fa-f]+}} <line:7:3, line:9:3> line:7:7 used getFirst 'int ()' implicit-inline
// CHECK-NEXT:`-CompoundStmt 0x{{[0-9A-Fa-f]+}} <col:18, line:9:3>
// CHECK-NEXT:`-ReturnStmt 0x{{[0-9A-Fa-f]+}} <line:8:4, col:16>
// CHECK-NEXT:`-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <col:11, col:16> 'int' <LValueToRValue>
// CHECK-NEXT:`-MemberExpr 0x{{[0-9A-Fa-f]+}} <col:11, col:16> 'int' lvalue .First 0x{{[0-9A-Fa-f]+}}
// CHECK-NEXT:`-CXXThisExpr 0x{{[0-9A-Fa-f]+}} <col:11> 'Pair' lvalue this
// CHECK-NEXT:-CXXMethodDecl 0x{{[0-9A-Fa-f]+}} <line:11:3, line:13:3> line:11:7 used getSecond 'int ()' implicit-inline
// CHECK-NEXT:`-CompoundStmt 0x{{[0-9A-Fa-f]+}} <col:19, line:13:3>
// CHECK-NEXT:`-ReturnStmt 0x{{[0-9A-Fa-f]+}} <line:12:5, col:12>
// CHECK-NEXT:`-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <col:12> 'int' <LValueToRValue>
// CHECK-NEXT:`-MemberExpr 0x{{[0-9A-Fa-f]+}} <col:12> 'int' lvalue .Second 0x{{[0-9A-Fa-f]+}}
// CHECK-NEXT:`-CXXThisExpr 0x{{[0-9A-Fa-f]+}} <col:12> 'Pair' lvalue implicit this


// CHECK:     CXXMethodDecl 0x{{[0-9A-Fa-f]+}} <line:19:3, line:21:3> line:19:7 used getSum 'int ()' implicit-inline
// CHECK-NEXT:`-CompoundStmt 0x{{[0-9A-Fa-f]+}} <col:16, line:21:3>
// CHECK-NEXT:`-ReturnStmt 0x{{[0-9A-Fa-f]+}} <line:20:5, col:25>
// CHECK-NEXT:`-BinaryOperator 0x{{[0-9A-Fa-f]+}} <col:12, col:25> 'int' '+'
// CHECK-NEXT:-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <col:12, col:17> 'int' <LValueToRValue>
// CHECK-NEXT:`-MemberExpr 0x{{[0-9A-Fa-f]+}} <col:12, col:17> 'int' lvalue .First 0x{{[0-9A-Fa-f]+}}
// CHECK-NEXT:`-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <col:12> 'Pair' lvalue <UncheckedDerivedToBase (Pair)>
// CHECK-NEXT:`-CXXThisExpr 0x{{[0-9A-Fa-f]+}} <col:12> 'PairInfo' lvalue this
// CHECK-NEXT:`-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <col:25> 'int' <LValueToRValue>
// CHECK-NEXT:`-MemberExpr 0x{{[0-9A-Fa-f]+}} <col:25> 'int' lvalue .Second 0x{{[0-9A-Fa-f]+}}
// CHECK-NEXT:`-ImplicitCastExpr 0x{{[0-9A-Fa-f]+}} <col:25> 'Pair' lvalue <UncheckedDerivedToBase (Pair)>
// CHECK-NEXT:`-CXXThisExpr 0x{{[0-9A-Fa-f]+}} <col:25> 'PairInfo' lvalue implicit this
