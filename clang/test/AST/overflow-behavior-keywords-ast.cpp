// RUN: %clang_cc1 -fexperimental-overflow-behavior-types -ast-dump %s | FileCheck %s

// Test that keyword and attribute syntax produce the same OverflowBehaviorType

// Attribute syntax
int __attribute__((overflow_behavior(wrap))) attr_wrap;
int __attribute__((overflow_behavior(trap))) attr_trap;

// Keyword syntax
int __ob_wrap keyword_wrap;
int __ob_trap keyword_trap;

// CHECK: VarDecl {{.*}} attr_wrap '__ob_wrap int'
// CHECK: VarDecl {{.*}} attr_trap '__ob_trap int'
// CHECK: VarDecl {{.*}} keyword_wrap '__ob_wrap int'
// CHECK: VarDecl {{.*}} keyword_trap '__ob_trap int'
