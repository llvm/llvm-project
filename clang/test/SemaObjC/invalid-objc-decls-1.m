// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface Super @end
Super s1; // expected-error{{interface type cannot be statically allocated}}

extern Super e1; // expected-error{{interface type cannot be statically allocated}}

struct S {
  Super s1; // expected-error{{interface type cannot be statically allocated}}
};

@protocol P1 @end

@interface INTF
{
  Super ivar1; // expected-error{{interface type cannot be statically allocated}}
}
@end

struct whatever {
  Super objField; // expected-error{{interface type cannot be statically allocated}}
};

@interface MyIntf
{
  Super<P1> ivar1; // expected-error{{interface type cannot be statically allocated}}
}
@end

Super foo( // expected-error{{interface type 'Super' cannot be returned by value; did you forget * in 'Super'}}
          Super parm1) { // expected-error{{interface type 'Super' cannot be passed by value; did you forget * in 'Super'}}
	Super p1; // expected-error{{interface type cannot be statically allocated}}
	return p1;
}

@interface NSMutableSet @end

@interface DVTDummyAnnotationProvider  
  @property(readonly) NSMutableSet annotations;	// expected-error{{interface type cannot be statically allocated}}

@end

