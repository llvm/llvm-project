// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -std=hlsl2015 -ast-dump %s | FileCheck %s --check-prefix=EARLIER
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -std=hlsl2016 -ast-dump %s | FileCheck %s --check-prefix=EARLIER
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -std=hlsl2017 -ast-dump %s | FileCheck %s --check-prefix=EARLIER
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -std=hlsl2018 -ast-dump %s | FileCheck %s --check-prefix=EARLIER
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -std=hlsl2021 -ast-dump %s | FileCheck %s --check-prefix=EARLIER
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -std=hlsl202x -DLATER -ast-dump %s | FileCheck %s --check-prefix=LATER
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -std=hlsl202y -DLATER -ast-dump %s | FileCheck %s --check-prefix=LATER

// LATER-NOT: UsingDirectiveDecl {{.*}} Namespace {{.*}} 'hlsl'
// LATER: ParmVarDecl {{.*}} Value 'hlsl::vector<float, 2>'

// EARLIER: UsingDirectiveDecl {{.*}} <<invalid sloc>> <invalid sloc> Namespace {{.*}} 'hlsl'
// EARLIER: ParmVarDecl {{.*}} Value 'vector<float, 2>'

#ifdef LATER
void test(hlsl::vector<float, 2> Value);
#else
void test(vector<float, 2> Value);
#endif
