// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface Test {
   int x;
}

-(void) setX: (int) d;
@end

extern struct foo x;

@implementation Test

-(void) setX: (int) n {
   x = n;
}

@end

@interface Ivar
- (float*)method;
@end

@interface A {
  A *Ivar;
}
- (int*)method;
@end

@implementation A
- (int*)method {
  int *ip = [Ivar method]; // expected-warning{{incompatible pointer types initializing 'int *' with an expression of type 'float *'}}
                           // Note that there is no warning in Objective-C++
  return 0;
}
@end

@interface TwoIvars {
  int a;
  int b;
}
@end

@implementation TwoIvars
+ (int)classMethod {
  return a + b; // expected-error{{instance variable 'a' accessed in class method}} \
  // expected-error{{instance variable 'b' accessed in class method}}
}
@end

@interface Radar10309454
{
  int IVAR; // expected-note 4 {{previous definition is here}}
}
@end

@interface Radar10309454()
{
  int IVAR; // expected-error {{instance variable is already declared}}
  int PIVAR; // expected-note {{previous definition is here}}
}
@end

@interface Radar10309454()
{
  int IVAR; // expected-error {{instance variable is already declared}}
}
@end

@interface Radar10309454()
{
  int IVAR; // expected-error {{instance variable is already declared}}
  int PIVAR; // expected-error {{instance variable is already declared}}
}
@end

@implementation Radar10309454
{
  int IVAR; // expected-error {{instance variable is already declared}}
}
@end

// PR5984
@interface Radar14037151 {
  int myStatus;
}
- (int) test;
@end

@implementation Radar14037151
- (int) test
{
  myStatus = 1;     // works
   __typeof(myStatus) __in;  // works.
  union U {
    __typeof(myStatus) __in;  // fails.
  };
  struct S {
    __typeof(myStatus) __in;  // fails.
    struct S1 { // expected-warning {{declaration does not declare anything}}
      __typeof(myStatus) __in;  // fails.
      struct S { // expected-warning {{declaration does not declare anything}}
        __typeof(myStatus) __in;  // fails.
      };
    };
  };

  return 0;
}
@end

@class NSString, NSData, NSNumber;

@interface NSObject
{
  Class isa;
}
@end

@interface Foo
{
  int a;
  NSString* b;
  NSData* c;
}
@end

@interface Bar : Foo
@end

@interface Bar () {
	NSString *q_strong;
	NSNumber *r_strong;
	int d; // expected-note {{previous definition is here}}
	NSString *e_strong; // expected-note {{previous definition is here}}
	NSData *f_weak; // expected-note {{previous definition is here}}
	int g; // expected-note 2 {{previous definition is here}}
}
@end

@interface Bar () {
	int g; // expected-note {{previous definition is here}} \
               // expected-error {{instance variable is already declared}}
}
@end

@implementation Bar {
	int d; // expected-error {{instance variable is already declared}}
	NSString *e_strong; // expected-error {{instance variable is already declared}}
	NSData *f_weak; // expected-error {{instance variable is already declared}}
	NSData *g; // expected-error 2 {{instance variable is already declared}}
}
@end
