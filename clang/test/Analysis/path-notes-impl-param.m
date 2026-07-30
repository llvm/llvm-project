// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.RetainCount -analyzer-output=text -verify %s

@protocol NSObject
@end

@interface NSObject <NSObject> {}
- (id)init;
+ (id)alloc;
- (id)autorelease;
@end

@interface Foo : NSObject
@property(nonatomic) int bar;
@end

@implementation Foo
-(int)bar {
  return 0;
}
@end

int baz() {
  Foo *f = [Foo alloc];
  // expected-note@-1 {{'f' initialized here}}
  // expected-note@-2 {{Method returns an instance of Foo with a +1 retain count}}

  return f.bar;
  // expected-warning@-1 {{Potential leak of an object stored into 'self' [osx.cocoa.RetainCount]}}
  // expected-note@-2 {{Passing value via implicit parameter 'self'}}
  // expected-note@-3 {{Object leaked: object allocated and stored into 'self' is not referenced later in this execution path and has a retain count of +1}}
}
