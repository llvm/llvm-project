// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface A {
  int X __attribute__((deprecated)); // expected-note 2 {{declared here}}
}
+ (void)F __attribute__((deprecated)); // expected-note 2 {{declared here}}
- (void)f __attribute__((deprecated)); // expected-note 4 {{declared here}}
@end

@implementation A
+ (void)F __attribute__((deprecated))
{
  [self F]; // no warning, since the caller is also deprecated.
}

- (void)g
{
  X++;        // expected-warning{{'X' is deprecated}}
  self->X++;  // expected-warning{{'X' is deprecated}}
  [self f]; // expected-warning{{'f' is deprecated}}
}

- (void)f
{
  [self f]; // no warning, the caller is deprecated in its interface.
}
@end

@interface B: A
@end
  
@implementation B
+ (void)G
{
  [super F]; // expected-warning{{'F' is deprecated}}
}

- (void)g
{
  [super f]; // // expected-warning{{'f' is deprecated}}
}
@end

@protocol P
- (void)p __attribute__((deprecated)); // expected-note {{declared here}}
@end

void t1(A *a)
{
  [A F]; // expected-warning{{'F' is deprecated}}
  [a f]; // expected-warning{{'f' is deprecated}}
}

void t2(id a)
{
  [a f];
}

void t3(A<P>* a)
{
  [a f]; // expected-warning{{'f' is deprecated}}
  [a p]; // expected-warning{{'p' is deprecated}}
} 

void t4(Class c)
{
  [c F];
}



@interface Bar 

@property (assign, setter = MySetter:) int FooBar __attribute__ ((deprecated)); // expected-note 2 {{declared here}}
- (void) MySetter : (int) value;
@end

int t5() {
  Bar *f;
  f.FooBar = 1;	   // expected-warning {{'FooBar' is deprecated}}
  return f.FooBar; // expected-warning {{'FooBar' is deprecated}}
}


__attribute ((deprecated))  
@interface DEPRECATED { // expected-note 2 {{declared here}}
  @public int ivar; 
  DEPRECATED *ivar2; // no warning.
} 
- (int) instancemethod;
- (DEPRECATED *) meth; // no warning.
@property  int prop; 
@end

@interface DEPRECATED (Category) // no warning.
- (DEPRECATED *) meth2; // no warning.
@end

@interface DEPRECATED (Category2) // no warning.
@end

@implementation DEPRECATED (Category2) // expected-warning {{'DEPRECATED' is deprecated}}
@end

@interface NS : DEPRECATED  // expected-warning {{'DEPRECATED' is deprecated}}
@end


@interface Test2
@property int test2 __attribute__((deprecated)); // expected-note 4 {{declared here}} \
						 // expected-note 2 {{property 'test2' is declared deprecated here}}
@end

void test(Test2 *foo) {
  int x;
  x = foo.test2; // expected-warning {{'test2' is deprecated}}
  x = [foo test2]; // expected-warning {{'test2' is deprecated}}
  foo.test2 = x; // expected-warning {{'test2' is deprecated}}
  [foo setTest2: x]; // expected-warning {{'setTest2:' is deprecated}}
}

__attribute__((deprecated))
@interface A(Blah) // expected-error{{attributes may not be specified on a category}}
@end


typedef struct {
	int x;
} footype __attribute((deprecated)); // expected-note 2 {{declared here}}

@interface foo {
	footype a; // expected-warning {{'footype' is deprecated}}
	footype b __attribute((deprecated));
}
@property footype c; // expected-warning {{'footype' is deprecated}}
@property footype d __attribute((deprecated));
@end
