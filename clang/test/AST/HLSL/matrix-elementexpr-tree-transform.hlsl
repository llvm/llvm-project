// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -std=hlsl202x \
// RUN:   -finclude-default-header -ast-dump -ast-dump-filter=get00 %s | FileCheck %s

template <typename T>
T get00(matrix<T, 2, 2> m) {
  return m._m00;
}

// Force instantiation for T=float.
float caller(matrix<float, 2, 2> m) {
  return get00(m);
}

// CHECK-LABEL: FunctionTemplateDecl {{.*}} get00
// CHECK: FunctionDecl {{.*}} get00 'T (matrix<T, 2, 2>)'
// CHECK-LABEL: FunctionDecl {{.*}} get00 'float (matrix<float, 2, 2>)'
// CHECK: MatrixElementExpr
