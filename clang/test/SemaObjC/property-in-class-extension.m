// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://7766184

@interface Foo @end

@interface Foo ()
  @property (readonly) int bar;
@end

void FUNC () {
    Foo *foo;
    foo.bar = 0; // expected-error {{assignment to readonly property}}
}

// rdar://8747333
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
- (NSObject *)warn;	// expected-note {{method definition for 'warn' not found}}
- (void)setWarn : (NSObject *)val; // expected-note {{method definition for 'setWarn:' not found}}
@end

@implementation rdar8747333 // expected-warning {{incomplete implementation}}
@synthesize bar = _bar;
@synthesize baz = _baz;
@synthesize bam = _bam;
@dynamic warn;
@end

