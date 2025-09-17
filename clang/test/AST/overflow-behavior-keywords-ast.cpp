// RUN: %clang_cc1 -foverflow-behavior-types -ast-dump %s | FileCheck %s

// Test that keyword and attribute syntax produce the same OverflowBehaviorType

// Attribute syntax
int __attribute__((overflow_behavior(wrap))) attr_wrap;
int __attribute__((overflow_behavior(no_wrap))) attr_no_wrap;

// Keyword syntax
int __wrap keyword_wrap;
int __no_wrap keyword_no_wrap;

// CHECK: VarDecl {{.*}} attr_wrap '__wrap int'
// CHECK: VarDecl {{.*}} attr_no_wrap '__no_wrap int'
// CHECK: VarDecl {{.*}} keyword_wrap '__wrap int'
// CHECK: VarDecl {{.*}} keyword_no_wrap '__no_wrap int'
