// RUN: %clang_cc1 -arcmt-check -verify -triple x86_64-apple-macosx10.7 %s

#include "Common.h"

@interface NSInvocation : NSObject
- (void)getReturnValue:(void *)retLoc;
- (void)setReturnValue:(void *)retLoc;

- (void)getArgument:(void *)argumentLocation atIndex:(int)idx;
- (void)setArgument:(void *)argumentLocation atIndex:(int)idx;
@end

@interface Test
@end

@implementation Test {
  id strong_id;
  __weak id weak_id;
  __unsafe_unretained id unsafe_id;
  int arg;
}
- (void) test:(NSInvocation *)invok {
  [invok getReturnValue:&strong_id]; // expected-error {{NSInvocation's getReturnValue is not safe to be used with an object with ownership other than __unsafe_unretained}}
  [invok getReturnValue:&weak_id]; // expected-error {{NSInvocation's getReturnValue is not safe to be used with an object with ownership other than __unsafe_unretained}}
  [invok getReturnValue:&unsafe_id];
  [invok getReturnValue:&arg];

  [invok setReturnValue:&strong_id]; // expected-error {{NSInvocation's setReturnValue is not safe to be used with an object with ownership other than __unsafe_unretained}}
  [invok setReturnValue:&weak_id]; // expected-error {{NSInvocation's setReturnValue is not safe to be used with an object with ownership other than __unsafe_unretained}}
  [invok setReturnValue:&unsafe_id];
  [invok setReturnValue:&arg];

  [invok getArgument:&strong_id atIndex:0]; // expected-error {{NSInvocation's getArgument is not safe to be used with an object with ownership other than __unsafe_unretained}}
  [invok getArgument:&weak_id atIndex:0]; // expected-error {{NSInvocation's getArgument is not safe to be used with an object with ownership other than __unsafe_unretained}}
  [invok getArgument:&unsafe_id atIndex:0];
  [invok getArgument:&arg atIndex:0];

  [invok setArgument:&strong_id atIndex:0]; // expected-error {{NSInvocation's setArgument is not safe to be used with an object with ownership other than __unsafe_unretained}}
  [invok setArgument:&weak_id atIndex:0]; // expected-error {{NSInvocation's setArgument is not safe to be used with an object with ownership other than __unsafe_unretained}}
  [invok setArgument:&unsafe_id atIndex:0];
  [invok setArgument:&arg atIndex:0];
}
@end
