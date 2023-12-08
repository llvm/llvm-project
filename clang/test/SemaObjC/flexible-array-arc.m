// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class -DNOARC %s
#ifdef NOARC
// expected-no-diagnostics
#endif

@interface RetainableArray {
  id flexible[];
#ifndef NOARC
  // expected-error@-2 {{ARC forbids flexible array members with retainable object type}}
#endif
}
@end
@implementation RetainableArray
@end

// Emit diagnostic only if have @implementation.
@interface RetainableArrayWithoutImpl {
  id flexible[];
}
@end

// With ARC flexible array member objects can be only __unsafe_unretained
@interface UnsafeUnretainedArray {
  __unsafe_unretained id flexible[];
}
@end
@implementation UnsafeUnretainedArray
@end

@interface NotObjCLifetimeTypeArray {
  char flexible[];
}
@end
@implementation NotObjCLifetimeTypeArray
@end
