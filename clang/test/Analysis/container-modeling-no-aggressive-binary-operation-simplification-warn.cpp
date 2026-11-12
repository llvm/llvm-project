// RUN: %clang_analyze_cc1 -std=c++11\
// RUN: -analyzer-checker=core,cplusplus,alpha.cplusplus.ContainerModeling\
// RUN: %s 2>&1 | FileCheck %s

// XFAIL: *

// CHECK: checker cannot be enabled with analyzer option 'aggressive-binary-operation-simplification' == false
