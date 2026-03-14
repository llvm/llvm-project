// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o -

typedef struct _NSPoint {
    float x;
    float y;
} NSPoint;

@interface Intf
- (void) MyMeth : (NSPoint) Arg1;
@end

@implementation Intf
- (void) MyMeth : (NSPoint) Arg1{}
@end

