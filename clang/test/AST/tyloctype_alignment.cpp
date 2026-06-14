// RUN: %clang_cc1 -ast-dump  %s | FileCheck %s

// Regression test for assertion failure in CreateTypeSourceInfo
// due to incorrect TypeLoc data size alignment.

// CHECK: FunctionTemplateDecl {{.*}} fpclassify
template<class T>
int fpclassify(T v);
template<>
int fpclassify<float> (float v);
