// RUN: %clang_cc1 -emit-llvm  -triple x86_64-apple-darwin -x objective-c %s -o - | FileCheck %s

@interface A {
        struct {
                unsigned char a : 1;
                unsigned char b : 1;
                unsigned char c : 1;
        } _flags;
}

@end

@implementation A

- (id)init {
        _flags.a = 1;
        _flags.b = 1;
        _flags.c = 1;

        return self;
}

@end

// CHECK: [[T1:%.*]] = load i64, ptr @"OBJC_IVAR_$_A._flags", align 8, !invariant.load ![[MD_NUM:[0-9]+]]
// CHECK: [[T2:%.*]] = load i64, ptr @"OBJC_IVAR_$_A._flags", align 8, !invariant.load ![[MD_NUM]]
// CHECK: [[T3:%.*]] = load i64, ptr @"OBJC_IVAR_$_A._flags", align 8, !invariant.load ![[MD_NUM]]
//
