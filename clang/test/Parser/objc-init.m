// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-fragile -verify -pedantic -Wno-objc-root-class %s
// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-fragile -verify -x objective-c++ -Wno-c99-designator -Wno-objc-root-class %s
// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-fragile -verify -x objective-c++ -Wno-c99-designator -Wno-objc-root-class -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -fobjc-runtime=macosx-fragile -verify -x objective-c++ -Wno-c99-designator -Wno-objc-root-class -std=c++11 %s
// rdar://5707001

@interface NSNumber;
- () METH;
- (unsigned) METH2;
@end

struct SomeStruct {
  int x, y, z, q;
};

void test1(void) {
	id objects[] = {[NSNumber METH]};
}

void test2(NSNumber x) { // expected-error {{interface type 'NSNumber' cannot be passed by value; did you forget * in 'NSNumber'}}
	id objects[] = {[x METH]};
}

void test3(NSNumber *x) {
	id objects[] = {[x METH]};
}


// rdar://5977581
void test4(void) {
  unsigned x[] = {[NSNumber METH2]+2};
}

void test5(NSNumber *x) {
  unsigned y[] = {
    [4][NSNumber METH2]+2,   // expected-warning {{use of GNU 'missing =' extension in designator}}
    [4][x METH2]+2   // expected-warning {{use of GNU 'missing =' extension in designator}}
  };
  
  struct SomeStruct z = {
    .x = [x METH2], // ok in C++98.
#if __cplusplus >= 201103L
    // expected-error@-2 {{non-constant-expression cannot be narrowed from type 'unsigned int' to 'int' in initializer list}}
    // expected-note@-3 {{insert an explicit cast to silence this issue}}
#endif
    .x [x METH2]    // expected-error {{expected '=' or another designator}}
#if __cplusplus >= 201103L
    // expected-error@-2 {{non-constant-expression cannot be narrowed from type 'unsigned int' to 'int' in initializer list}}
    // expected-note@-3 {{insert an explicit cast to silence this issue}}
#endif
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



