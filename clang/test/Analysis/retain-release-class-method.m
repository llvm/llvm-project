// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,osx.cocoa.RetainCount -verify %s

@interface NSObject
+ (id)class;
+ (id)alloc;
- (id)init;
- (void)release;
- (id)performSelector:(SEL)aSelector;
@end

void testClassMethodRetained() {
  id x = [NSObject class];
  id y = [x alloc]; // expected-warning{{Potential leak of an object}}
}
