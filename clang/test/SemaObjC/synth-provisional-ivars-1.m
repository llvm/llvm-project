// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics
// rdar://8913053

typedef unsigned char BOOL;

@interface MailApp
{
  BOOL _isAppleInternal;
}
@property(assign) BOOL isAppleInternal;
@end

static BOOL isAppleInternal(void) {return 0; }

@implementation MailApp

- (BOOL)isAppleInternal {
    return _isAppleInternal;
}

- (void)setIsAppleInternal:(BOOL)flag {
    _isAppleInternal= !!flag;
}

- (void) Meth {
    self.isAppleInternal = isAppleInternal();
}
@end
