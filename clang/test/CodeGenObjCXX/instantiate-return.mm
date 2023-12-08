// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -std=c++11 -o - %s | FileCheck %s

template <class T>
struct TemplateClass {
  int a = 0;
};

struct S0;

@interface C1
- (TemplateClass<S0>)m1;
@end

// This code used to assert in CodeGen because the return type TemplateClass<S0>
// wasn't instantiated.

// CHECK: define internal i32 @"\01-[C1 m1]"(

@implementation C1
- (TemplateClass<S0>)m1 {
}
@end
