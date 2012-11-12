// RUN: %clang_cc1 -fblocks -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fblocks -fsyntax-only -x objective-c %s > %t
// RUN: diff %t %s.result

#include "Common.h"

id IhaveSideEffect();

@interface Foo : NSObject {
  id bar;
}
@property (retain) id bar;
-(id)test:(id)obj;
-(id)something;
@end

#define Something_Macro(key, comment) \
 [[Foo new] something]

@implementation Foo

@synthesize bar;

-(id)something {}

-(id)test:(id)obj {
  id x = self.bar;
  [x retain];
  self.bar = obj;
  if (obj)
    [obj retain];

  [Something_Macro(@"foo", "@bar") retain];

  [IhaveSideEffect() retain];

  [[self something] retain];

  [[self retain] something];

  [[IhaveSideEffect() retain] release];
  [[x retain] release];
  // do stuff with x;
  [x release];
  return [self retain];
}
  
- (id)test1 {
  id x=0;
  ([x retain]);
  return ((([x retain])));
}
@end

id foo (Foo *p) {
    p = [p retain];
    return ([p retain]);
}

void block_tests(Foo *p) {
  id (^B)() = ^() {
    if (p) {
      id (^IB)() = ^() {
        id bar = [p retain];
        return bar;
      };
      IB();
    }
    return [p retain];
  };
}
