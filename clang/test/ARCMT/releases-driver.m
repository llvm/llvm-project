// RUN: %clang_cc1 -fblocks -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: cp %s %t
// RUN: %clang_cc1 -arcmt-modify -triple x86_64-apple-macosx10.6 -x objective-c %t
// RUN: diff %t %s.result
// RUN: rm %t

typedef int BOOL;

id IhaveSideEffect();

@protocol NSObject
- (BOOL)isEqual:(id)object;
- (id)retain;
- (oneway void)release;
@end

@interface NSObject <NSObject> {}
@end

@interface Foo : NSObject {
  id bar;
}
@property (retain) id bar;
-(void)test:(id)obj;
@end

@implementation Foo

@synthesize bar;

-(void)test:(id)obj {
  id x = self.bar;
  [x retain];
  self.bar = obj;
  // do stuff with x;
  [x release];

  [IhaveSideEffect() release];

  [x release], x = 0;
}
  
@end

void func(Foo *p) {
  [p release];
  (([p release]));
}

@interface Baz {
	id <NSObject> _foo;
}
@end

@implementation Baz
- (void) dealloc {
  [_foo release];
}
@end

#define RELEASE_MACRO(x) [x release]
#define RELEASE_MACRO2(x) RELEASE_MACRO(x)

void test2(id p) {
  RELEASE_MACRO(p);
  RELEASE_MACRO2(p);
}
