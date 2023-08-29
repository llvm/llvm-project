// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5  -emit-llvm -o - %s | FileCheck -check-prefix CHECK-LP64 %s

@interface Foo 
- (id)test:(id)one, id two;
@end

@implementation Foo
- (id)test:(id )one, id two {return two; } @end

// CHECK-LP64: private unnamed_addr constant [11 x i8] c"@24@0:8@16
