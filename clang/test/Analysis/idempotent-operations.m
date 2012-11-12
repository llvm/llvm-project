// RUN: %clang_cc1 -analyze -analyzer-store=region -analyzer-constraints=range -fblocks -analyzer-opt-analyze-nested-blocks -analyzer-checker=alpha.deadcode.IdempotentOperations,osx.cocoa.RetainCount -verify %s
// expected-no-diagnostics

typedef signed char BOOL;
typedef unsigned long NSUInteger;
typedef struct _NSZone NSZone;
@protocol NSObject  - (BOOL)isEqual:(id)object;
@end

@interface NSObject {}
  @property int locked;
  @property(nonatomic, readonly) NSObject *media;
@end

// <rdar://problem/8725041> - Don't flag idempotent operation warnings when
// a method may invalidate an instance variable.
@interface Rdar8725041 : NSObject {
  id _attribute;
}
  - (void) method2;
@end

@implementation Rdar8725041
- (BOOL) method1 {
  BOOL needsUpdate = (BOOL)0;
  id oldAttribute = _attribute;
  [self method2];
  needsUpdate |= (_attribute != oldAttribute); // no-warning
  return needsUpdate;
}

- (void) method2
{
  _attribute = ((void*)0);
}
@end

// Test that the idempotent operations checker works in the prescence
// of property expressions.
void pr9116(NSObject *placeholder) {
  int x = placeholder.media.locked = placeholder ? 1 : 0;
}

// <rdar://problem/9130239>: Test that calling property setters doesn't 
// trigger an assertion failure when the object is nil.
@interface RDar9130239
@property (assign) id delegate;
@end

void test_RDar9130239(RDar9130239 *x) {
  if (x)
    return;
  x.delegate = x; // no-warning
}

