// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c %s > %t
// RUN: diff %t %s.result

#include "Common.h"

@interface myController : NSObject
-(id)test:(id)x;
@end

#define MY_MACRO1(x)
#define MY_MACRO2(x) (void)x

@implementation myController
-(id) test:(id) x {
  [[x retain] release];
  return [[x retain] autorelease];
}

-(void)dealloc
{
  id array, array_already_empty;
  for (id element in array_already_empty) {
  }

  [array release];
  ;

  int b, b_array_already_empty;
  if (b)
    [array release];
  if (b_array_already_empty) ;

  if (b) {
    [array release];
  }
  if (b_array_already_empty) {
  }

  if (b)
    MY_MACRO1(array);
  if (b)
    MY_MACRO2(array);
}
@end
