// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-darwin10  -fblocks -verify -Wno-objc-root-class %s 
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -triple x86_64-apple-darwin10  -fblocks -verify -Wno-objc-root-class %s 
// expected-no-diagnostics

@interface Blocky @end

@implementation Blocky {
    int _a;
}
- (int)doAThing {
    ^{
        char self;
        return _a;
    }();
    return _a;
}

@end

@interface ShadowSelf
{
    int _anIvar;
}
@end

@interface C {
  int _cIvar;
}
@end

@implementation ShadowSelf 
- (void)doSomething {
    __typeof(self) newSelf = self;
    {
        __typeof(self) self = newSelf;
        (void)_anIvar;
    }
    {
      C* self;	
      (void) _anIvar;
    }
}
- (void)doAThing {
    ^{
        id self;
	(void)_anIvar;
    }();
}
@end

