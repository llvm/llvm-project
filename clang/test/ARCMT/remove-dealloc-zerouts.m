// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c %s > %t
// RUN: diff %t %s.result

@interface Foo 
@property (retain) id x;
@property (retain) id y;
@property (retain) id w;
@property (retain) id z;
@property (strong) id q;
@end

@implementation Foo 
@synthesize x;
@synthesize y;
@synthesize w;
@synthesize q;
@dynamic z;

- (void) dealloc {
  self.x = self.y = self.w = 0;
  self.x = 0, w = 0, y = 0;
  [self setY:0];
  w = 0;
  q = 0;
  self.z = 0;
}
@end

@interface Bar
@property (retain) Foo *a;
- (void) setA:(Foo*) val;
- (id) a;
@end

@implementation Bar
- (void) dealloc {
  [self setA:0];  // This is user-defined setter overriding synthesize, don't touch it.
  self.a.x = 0;  // every dealloc must zero out its own ivar. This patter is not recognized.
}
@synthesize a;
- (void) setA:(Foo*) val { }
- (id) a {return 0;}
@end
