// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx -verify %s

typedef signed char BOOL;

@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
+(id)new;
-(id)init;
-(id)autorelease;
-(id)copy;
- (Class)class;
-(id)retain;
- (oneway void)release;
@end

@interface Cell : NSObject {
  int x;
}
- (id) init;
- (void)test;
@end

@implementation Cell
- (id) init {
  if ((self = [super init])) {
    return self;
  }
  // Test that this is being analyzed.
  int m;
  m = m + 1; //expected-warning {{The left operand of '+' is a garbage value}}
  return self;
}

// Make sure that we do not propagate the 'nil' check from inlined 'init' to 'test'.
- (void) test {
  Cell *newCell = [[Cell alloc] init];
  newCell->x = 5; // no-warning
  [newCell release];
}
@end
