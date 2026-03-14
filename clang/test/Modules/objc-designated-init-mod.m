// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/objc-desig-init %s -verify
// expected-no-diagnostics

#import "X.h"
#import "Base.h"
#import "A.h"

@implementation X

- (instancetype)initWithNibName:(NSString *)nibName {
  if ((self = [super initWithNibName:nibName])) {
		return self;
  }
  return self;
}
@end
