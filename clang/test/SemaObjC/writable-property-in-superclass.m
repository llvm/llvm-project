// RUN: %clang_cc1  -fsyntax-only -verify %s
// expected-no-diagnostics

@interface MessageStore
@property (assign, readonly) int P;
@end

@interface MessageStore (CAT)
@property (assign) int P;
@end

@interface  NeXTMbox : MessageStore
@end

@implementation NeXTMbox
- (void) Meth { self.P = 1; }
@end

