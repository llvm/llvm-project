// RUN: %clang_cc1 -triple x86_64-apple-macosx10.6 -fsyntax-only -fobjc-runtime-has-weak -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-macosx10.6 -fsyntax-only -fobjc-gc-only -x objective-c %s > %t
// RUN: diff %t %s.result
// RUN: arcmt-test --args -triple x86_64-apple-macosx10.6 -fsyntax-only -fobjc-gc-only -x objective-c++ %s > %t
// RUN: diff %t %s.result

// MRC __weak broke this test somehow.
// XFAIL: *

#include "Common.h"
#include "GC.h"

void test1(CFTypeRef *cft) {
  id x = NSMakeCollectable(cft);
}

@interface I1
@end

@implementation I1
-(void)dealloc {
  // dealloc
  test1(0);
}

-(void)finalize {
  // finalize
  test1(0);
}
@end

@interface I2
@property (retain) id prop;
@end

@implementation I2
@synthesize prop;

-(void)finalize {
  self.prop = 0;
  // finalize
  test1(0);
}
@end

__attribute__((objc_arc_weak_reference_unavailable))
@interface QQ {
  __weak id s;
  __weak QQ *q;
}
@end

@interface I3
@property (assign) I3 *__weak pw1, *__weak pw2;
@property (assign) I3 *__strong ps;
@property (assign) I3 * pds;
@end

@interface I4Impl {
  I4Impl *pds2;
}
@property (assign) I4Impl *__weak pw1, *__weak pw2;
@property (assign) I4Impl *__strong ps;
@property (assign) I4Impl * pds;
@property (assign) I4Impl * pds2;
@end

@implementation I4Impl
@synthesize pw1, pw2, ps, pds, pds2;

-(void)test1:(CFTypeRef *)cft {
  id x = NSMakeCollectable(cft);
}
@end

@interface I5 {
  __weak id prop;
}
@property (readonly) __weak id prop;
@end
