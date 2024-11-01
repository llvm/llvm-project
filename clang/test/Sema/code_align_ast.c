// RUN: %clang_cc1 -fsyntax-only -ast-dump -verify -x c %s | FileCheck -check-prefix=CHECK-C %s
// RUN: %clang_cc1 -fsyntax-only -ast-dump -x c++ -std=c++11 %s | FileCheck %s --check-prefixes CHECK-C,CHECK-CPP

// expected-no-diagnostics

void bar(int);
// CHECK-C: FunctionDecl{{.*}}code_align 'void ()'
void code_align() {
  int a1[10], a2[10];
  // CHECK-C: AttributedStmt
  // CHECK-C-NEXT: CodeAlignAttr
  // CHECK-C-NEXT:  ConstantExpr{{.*}}'int'
  // CHECK-C-NEXT:  value: Int 16
  // CHECK-C-NEXT:  IntegerLiteral{{.*}}16{{$}}
  [[clang::code_align(16)]] for (int p = 0; p < 128; ++p) {
    a1[p] = a2[p] = 0;
  }

  // CHECK-C: AttributedStmt
  // CHECK-C-NEXT:  CodeAlignAttr
  // CHECK-C-NEXT:  ConstantExpr{{.*}}'int'
  // CHECK-C-NEXT:  value: Int 4
  // CHECK-C-NEXT:  IntegerLiteral{{.*}}4{{$}}
  int i = 0;
  [[clang::code_align(4)]] while (i < 30) {
    a1[i] += 3;
  }

  // CHECK-C: AttributedStmt
  // CHECK-C-NEXT:  CodeAlignAttr
  // CHECK-C-NEXT:  ConstantExpr{{.*}}'int'
  // CHECK-C-NEXT:  value: Int 32
  // CHECK-C-NEXT:  IntegerLiteral{{.*}}32{{$}}
  for (int i = 0; i < 128; ++i) {
    [[clang::code_align(32)]]  for (int j = 0; j < 128; ++j) {
      a1[i] += a1[j];
    }
  }

  // CHECK-C: AttributedStmt
  // CHECK-C-NEXT:  CodeAlignAttr
  // CHECK-C-NEXT:  ConstantExpr{{.*}}'int'
  // CHECK-C-NEXT:  value: Int 64
  // CHECK-C-NEXT:  IntegerLiteral{{.*}}64{{$}}
  [[clang::code_align(64)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // CHECK-C: AttributedStmt
  // CHECK-C-NEXT: CodeAlignAttr
  // CHECK-C-NEXT:  ConstantExpr{{.*}}'int'
  // CHECK-C-NEXT:  value: Int 4
  // CHECK-C-NEXT:  IntegerLiteral{{.*}}4{{$}}
  int b = 10;
  [[clang::code_align(4)]] do {
    b = b + 1;
  } while (b < 20);
}

#if __cplusplus >= 201103L
//CHECK-CPP: FunctionDecl{{.*}}used code_align_cpp 'void ()' implicit_instantiation
template <int A, int B>
void code_align_cpp() {
  int a[10];
  // CHECK-CPP: AttributedStmt
  // CHECK-CPP-NEXT:  CodeAlignAttr
  // CHECK-CPP-NEXT:  ConstantExpr{{.*}}'int'
  // CHECK-CPP-NEXT:  value: Int 32
  // CHECK-CPP-NEXT:  SubstNonTypeTemplateParmExpr{{.*}}'int'
  // CHECK-CPP-NEXT:  NonTypeTemplateParmDecl{{.*}}referenced 'int' depth 0 index 0 A
  // CHECK-CPP-NEXT:  IntegerLiteral{{.*}}32{{$}}
  [[clang::code_align(A)]] for (int i = 0; i != 10; ++i)
    a[i] = 0;

  // CHECK-CPP: AttributedStmt
  // CHECK-CPP-NEXT:  CodeAlignAttr
  // CHECK-CPP-NEXT:  ConstantExpr{{.*}}'int'
  // CHECK-CPP-NEXT:  value: Int 4
  // CHECK-CPP-NEXT:  SubstNonTypeTemplateParmExpr{{.*}}'int'
  // CHECK-CPP-NEXT:  NonTypeTemplateParmDecl{{.*}}referenced 'int' depth 0 index 1 B
  // CHECK-CPP-NEXT:  IntegerLiteral{{.*}}4{{$}}
  int c[] = {0, 1, 2, 3, 4, 5};
  [[clang::code_align(B)]] for (int n : c) { n *= 2; }
}

int main() {
  code_align_cpp<32, 4>();
  return 0;
}
#endif
