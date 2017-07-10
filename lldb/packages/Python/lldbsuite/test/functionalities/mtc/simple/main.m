#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>

int main() {
  NSView *view = [[NSView alloc] init];
  dispatch_group_t g = dispatch_group_create();
  dispatch_group_enter(g);
  [NSThread detachNewThreadWithBlock:^{
    @autoreleasepool {
      [view superview];
    }
    dispatch_group_leave(g);
  }];
  dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
}
