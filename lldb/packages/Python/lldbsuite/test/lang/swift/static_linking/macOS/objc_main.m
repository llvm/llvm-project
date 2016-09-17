#import <Foundation/Foundation.h>

#import "A-Swift.h"
#import "B-Swift.h"

int main(int argc, const char * argv[]) {
  @autoreleasepool {
      NSLog(@"Hello, World!");
      NSLog(@"A = %ld", [[[A alloc] init] foo]);
      NSLog(@"B = %ld", [[[B alloc] init] bar]);
  }
	return 0;
}
