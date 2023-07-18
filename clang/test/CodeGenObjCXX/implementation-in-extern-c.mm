// RUN: %clang_cc1 -emit-llvm %s -o /dev/null
// rdar://12581683

extern "C" {
@interface RetainBucket 
+ (id) sharedRetainBucket;
@end

@implementation RetainBucket
+ (id) sharedRetainBucket
{
    static id sharedBucket = (id)0;
    return sharedBucket;
}
@end
}

