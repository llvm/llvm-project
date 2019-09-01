#import <Foundation/Foundation.h>

namespace error {
int blah;
}

@interface Util : NSObject
+ (void)debugPrintErrorStatic;
- (void)debugPrintError;
@end

@implementation Util
+ (void)debugPrintErrorStatic {
  NSError* error = [NSError errorWithDomain:NSURLErrorDomain code:-1 userInfo:nil];
  NSLog(@"xxx, error = %@", error); // break here
}

- (void)debugPrintError {
  NSError* error = [NSError errorWithDomain:NSURLErrorDomain code:-1 userInfo:nil];
  NSLog(@"xxx, error = %@", error); // break here
}
@end
