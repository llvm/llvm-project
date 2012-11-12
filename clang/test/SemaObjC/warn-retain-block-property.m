// RUN: %clang_cc1 -fsyntax-only -fblocks -fobjc-arc -verify -Wno-objc-root-class %s
// rdar://9829425
// RUN: %clang_cc1 -fsyntax-only -fblocks -verify -Wno-objc-root-class %s
// rdar://11761511

extern void doSomething();

@interface Test
{
@public
  void (^aBlock)(void);
}
@property (retain) void (^aBlock)(void); // expected-warning {{retain'ed block property does not copy the block - use copy attribute instead}}
@property (weak, retain) void (^aBlockW)(void); // expected-error {{property attributes 'retain' and 'weak' are mutually exclusive}} 
@property (strong, retain) void (^aBlockS)(void); // OK
@property (readonly, retain) void (^aBlockR)(void); // OK
@property (copy, retain) void (^aBlockC)(void); // expected-error {{property attributes 'copy' and 'retain' are mutually exclusive}}
@property (assign, retain) void (^aBlockA)(void); // expected-error {{property attributes 'assign' and 'retain' are mutually exclusive}}
@end

@implementation Test
@synthesize aBlock;
@dynamic aBlockW, aBlockS, aBlockR, aBlockC, aBlockA;
@end

int main() {
  Test *t;
  t.aBlock = ^{ doSomething(); };
  t.aBlockW = ^{ doSomething(); };
  t.aBlockS = ^{ doSomething(); };
}

