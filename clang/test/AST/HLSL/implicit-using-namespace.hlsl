// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -std=hlsl2015 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -std=hlsl2016 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -std=hlsl2017 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -std=hlsl2018 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -std=hlsl2021 -ast-dump %s | FileCheck %s

// CHECK: UsingDirectiveDecl {{.*}} <<invalid sloc>> <invalid sloc> Namespace {{.*}} 'hlsl'
// CHECK: ParmVarDecl {{.*}} Value 'vector<float, 2>'
void test(vector<float, 2> Value);
