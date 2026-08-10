// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s

@interface Test
- (void)test;
@end

@implementation Test
- (void)test __attribute__((stdcall)) {}
    // CHECK: define{{.*}}x86_stdcallcc{{.*}}Test test
    
- (void)test2 __attribute__((ms_abi)) {}
    // CHECK: define{{.*}}win64cc{{.*}}Test test2
@end
