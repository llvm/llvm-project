// RUN: %clang_cc1 -fsyntax-only -verify -Wparentheses -Wno-objc-root-class %s

// Don't warn about some common ObjC idioms unless we have -Widiomatic-parentheses on.
// <rdar://problem/7382435>

@interface Object 
{
  unsigned uid;
}
- (id) init;
- (id) initWithInt: (int) i;
- (id) myInit __attribute__((objc_method_family(init)));
- (void) iterate: (id) coll;
- (id) nextObject;
@property unsigned uid;
@end

@implementation Object
@synthesize uid;
- (id) init {
  if (self = [self init]) {
  }
  return self;
}

- (id) initWithInt: (int) i {
  if (self = [self initWithInt: i]) {
  }
  // rdar://11066598
  if (self.uid = 100) { // expected-warning {{using the result of an assignment as a condition without parentheses}} \
                        // expected-note {{place parentheses around the assignment to silence this warning}} \
                        // expected-note {{use '==' to turn this assignment into an equality comparison}}
        // ...
  }
  return self;
}

- (id) myInit {
  if (self = [self myInit]) {
  }
  return self;
}

- (void) iterate: (id) coll {
  id cur;
  while (cur = [coll nextObject]) {
  }
}

- (id) nextObject {
  return self;
}
@end
