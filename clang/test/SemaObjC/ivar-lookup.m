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

// rdar://10309454
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
