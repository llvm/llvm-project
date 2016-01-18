// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-fragile -verify -pedantic -Wno-objc-root-class %s
// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-fragile -verify -x objective-c++ -Wno-objc-root-class %s
// rdar://5707001

@interface NSNumber;
- () METH;
- (unsigned) METH2;
@end

struct SomeStruct {
  int x, y, z, q;
};

void test1() {
	id objects[] = {[NSNumber METH]};
}

void test2(NSNumber x) { // expected-error {{interface type 'NSNumber' cannot be passed by value; did you forget * in 'NSNumber'}}
	id objects[] = {[x METH]};
}

void test3(NSNumber *x) {
	id objects[] = {[x METH]};
}


// rdar://5977581
void test4() {
  unsigned x[] = {[NSNumber METH2]+2};
}

void test5(NSNumber *x) {
  unsigned y[] = {
    [4][NSNumber METH2]+2,   // expected-warning {{use of GNU 'missing =' extension in designator}}
    [4][x METH2]+2   // expected-warning {{use of GNU 'missing =' extension in designator}}
  };
  
  struct SomeStruct z = {
    .x = [x METH2], // ok.
    .x [x METH2]    // expected-error {{expected '=' or another designator}}
  };
}

// rdar://7370882
@interface SemicolonsAppDelegate 
{
  id i;
}
@property (assign) id window;
@end

@implementation SemicolonsAppDelegate
{
  id i;
}
  @synthesize window=i;
@end



