// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o -

@class NSString;

@interface NSConstantString;
@end



NSConstantString *t0 = @"123";
NSConstantString *t = @"123"     @"4567"; // concat
NSConstantString *t1 = @"123"     @"4567" /* COMMENT */ @"89"; // concat
NSConstantString *t2 = @"123"     @/* COMMENT */ "4567"; // concat

