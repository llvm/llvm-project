// RUN: cp %s %t
// RUN: %clang_cc1 -x objective-c -fixit %t
// RUN: %clang_cc1 -x objective-c -Werror %t
//rdar://17911746

@class BridgeFormatter;

@interface NSObject 
+ (id)new;
@end

@interface X : NSObject
@property int x;
@property int Y;
@property(assign, readwrite, getter=formatter, setter=setFormatter:) BridgeFormatter* cppFormatter;
@end

@implementation X
- (void) endit
{
 self.formatter = 0;
}
@end
 
int main(void)
{
    X *obj = [X new];
    obj.X = 3;
    obj.y = 4;
    return obj.x + obj.Y;
}
