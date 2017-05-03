#import "TestExt.h"
#import "Foo.h"

@implementation Test (Stuff)
- (void)doSomethingElse: (CMTimeRange *)range_ptr {
    NSLog(@"doSomethingElse: %p", range_ptr); // break here
}
@end
