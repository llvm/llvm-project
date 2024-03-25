// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@interface I {
}

@property int IVAR; 
- (int) OK;
@end

@implementation I
- (int) Meth { return _IVAR; }
- (int) OK { return self.IVAR; }
@end
