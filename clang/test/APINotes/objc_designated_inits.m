// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -iapinotes-modules %S/Inputs/APINotes -fapinotes-cache-path=%t/APINotesCache -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -verify

#include "HeaderLib.h"
#import <SomeKit/SomeKit.h>

@interface CSub : C
-(instancetype)initWithA:(A*)a;
@end

@implementation CSub
-(instancetype)initWithA:(A*)a { // expected-warning{{designated initializer missing a 'super' call to a designated initializer of the super class}}
  // expected-note@SomeKit/SomeKit.h:20 2{{method marked as designated initializer of the class here}}
  self = [super init]; // expected-warning{{designated initializer invoked a non-designated initializer}}
  return self;
}
@end
