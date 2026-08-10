// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o -

typedef struct NSMethodFrameArgInfo {
    struct NSMethodFrameArgInfo *subInfo;
    struct NSMethodFrameArgInfo *an;
} NSMethodFrameArgInfo;

@interface NSMethodSignature 
- (NSMethodFrameArgInfo *)_argInfo;
@end

@implementation NSMethodSignature

- (NSMethodFrameArgInfo *)_argInfo{
    return 0;
}

@end

