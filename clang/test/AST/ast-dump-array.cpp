// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s | FileCheck -strict-whitespace %s

void testArrayInitExpr()
{
    int a[10];
    auto l = [a]{
    };
    // CHECK: |-ArrayInitLoopExpr 0x{{[^ ]*}} <col:15> 'int [10]'
    // CHECK: |     `-ArrayInitIndexExpr 0x{{[^ ]*}} <<invalid sloc>> 'unsigned long'
}

template<typename T, int Size>
class array {
  T data[Size];

  using array_T_size = T[Size];
  // CHECK: `-DependentSizedArrayType 0x{{[^ ]*}} 'T [Size]' dependent   <col:25, col:30>
};

