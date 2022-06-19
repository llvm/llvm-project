// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10  -emit-llvm -o - %s | FileCheck %s
// rdar://11515196

@interface NSArray @end

@interface NSMutableArray : NSArray
- (void) addObject;
@end

@interface BPXLAppDelegate

- (NSArray *)arrayOfThings;

@end


@interface BPXLAppDelegate ()
@property (retain, nonatomic) NSMutableArray *arrayOfThings;
@end

@implementation BPXLAppDelegate

@synthesize arrayOfThings=_arrayOfThings;

- (void)applicationDidFinishLaunching
{
   [self.arrayOfThings addObject];
}

@end

// CHECK: define internal [[RET:%.*]]* @"\01-[BPXLAppDelegate arrayOfThings
// CHECK: [[THREE:%.*]] = bitcast [[OPQ:%.*]]* [[TWO:%.*]] to [[RET]]*
// CHECK: ret [[RET]]* [[THREE]]

