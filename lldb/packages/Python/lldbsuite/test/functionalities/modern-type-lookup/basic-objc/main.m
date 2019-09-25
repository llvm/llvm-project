#import <Foundation/Foundation.h>

@interface Foo : NSObject
+(int) bar: (int) string;
@end

@implementation Foo
+(int) bar: (int) x
{
  return x + x;
}
@end

int main() {
  NSLog(@"Hello World");
  return 0; // break here
}
