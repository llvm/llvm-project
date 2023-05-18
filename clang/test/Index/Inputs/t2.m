#include "objc.h"

static void foo() {
  Sub *sub;
  int x = [sub my_var];
  [sub my_method:x];
  [Sub my_method:x];
}

@implementation Sub
-(void) my_method: (int)param {
}
@end
