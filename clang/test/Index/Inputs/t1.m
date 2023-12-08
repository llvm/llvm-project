#include "objc.h"

static void foo() {
  Base *base;
  int x = [base my_var];
  [base my_method:x];
  [Base my_method:x];
}

@implementation Base
-(int) my_var {
  return my_var;
}

-(void) my_method: (int)param {
}

+(void) my_method: (int)param {
}
@end
