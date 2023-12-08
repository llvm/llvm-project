// RUN: %clang_cc1 -triple x86_64-apple-macosx10.7 -fsyntax-only -fobjc-runtime-has-weak -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-macosx10.7 -fsyntax-only -fobjc-gc-only -no-finalize-removal -x objective-c %s > %t
// RUN: diff %t %s.result
// RUN: arcmt-test --args -triple x86_64-apple-macosx10.7 -fsyntax-only -fobjc-gc-only -no-finalize-removal -x objective-c++ %s > %t
// RUN: diff %t %s.result

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
  I4Impl *pds3;
  __weak I4Impl *pw3;
  __weak I4Impl *pw4;
}
@property (assign) I4Impl *__weak pw1, *__weak pw2;
@property (assign) I4Impl *__strong ps;
@property (assign) I4Impl * pds;
@property (assign) I4Impl * pds2;
@property (readwrite) I4Impl * pds3;
@property (readonly) I4Impl * pds4;
@property (readonly) __weak I4Impl *pw3;
@property (assign) __weak I4Impl *pw4;
@end

@implementation I4Impl
@synthesize pw1, pw2, pw3, pw4, ps, pds, pds2, pds3, pds4;

-(void)test1:(CFTypeRef *)cft {
  id x = NSMakeCollectable(cft);
}
@end

@interface rdar10532449
@property (assign) id assign_prop;
@property (assign, readonly) id __strong strong_readonly_prop;
@property (assign) id __weak weak_prop;
@end

@implementation rdar10532449
@synthesize assign_prop, strong_readonly_prop, weak_prop;
@end
