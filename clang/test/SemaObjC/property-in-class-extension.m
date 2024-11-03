// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s

@interface Foo @end

@interface Foo ()
  @property (readonly) int bar;
@end

void FUNC (void) {
    Foo *foo;
    foo.bar = 0; // expected-error {{assignment to readonly property}}
}

@class NSObject;

@interface rdar8747333  {
@private
    NSObject *_bar;
    NSObject *_baz;
    NSObject *_bam;
}
- (NSObject *)baz;
@end

@interface rdar8747333 ()
- (NSObject *)bar;
@end

@interface rdar8747333 ()
@property (readwrite, assign) NSObject *bar;
@property (readwrite, assign) NSObject *baz;
@property (readwrite, assign) NSObject *bam;
@property (readwrite, assign) NSObject *warn;
@end

@interface rdar8747333 ()
- (NSObject *)bam;
- (NSObject *)warn;
- (void)setWarn : (NSObject *)val;
@end

@implementation rdar8747333
@synthesize bar = _bar;
@synthesize baz = _baz;
@synthesize bam = _bam;
@dynamic warn;
@end

