// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

// <rdar://problem/6463729>
@class XX;

void func() {
  XX *obj;
  void *vv;

  obj = vv; // expected-error{{assigning to 'XX *' from incompatible type 'void *'}}
}

// <rdar://problem/7952457>
@interface I
{
  void* delegate;
}
- (I*) Meth;
- (I*) Meth1;
@end

@implementation I 
- (I*) Meth { return static_cast<I*>(delegate); }
- (I*) Meth1 { return reinterpret_cast<I*>(delegate); }
@end

