// RUN: %clang_cc1 -verify -fsyntax-only -Wno-objc-root-class %s

@class I0; // expected-note 2{{forward declaration of class here}}

// rdar://6811884
int g0 = sizeof(I0); // expected-error{{invalid application of 'sizeof' to an incomplete type 'I0'}}

// rdar://6821047
void *g3(I0 *P) {
  P = P+5;        // expected-error {{arithmetic on a pointer to an incomplete type 'I0'}}

  return &P[4];   // expected-error{{expected method to read array element not found on object of type 'I0 *'}}
}



@interface I0 {
@public
  char x[4];
}

@property int p0;
@end

// size == 4
int g1[ sizeof(I0)     // expected-error {{application of 'sizeof' to interface 'I0' is not supported on this architecture and platform}}
       == 4 ? 1 : -1];

@implementation I0
@synthesize p0 = _p0;
@end

// size == 4 (we do not include extended properties in the
// sizeof).
int g2[ sizeof(I0)   // expected-error {{application of 'sizeof' to interface 'I0' is not supported on this architecture and platform}}
       == 4 ? 1 : -1];

@interface I1
@property int p0;
@end

@implementation I1
@synthesize p0 = _p0;
@end

typedef struct { @defs(I1); } I1_defs; // expected-error {{use of @defs is not supported on this architecture and platform}}

// FIXME: This is currently broken due to the way the record layout we
// create is tied to whether we have seen synthesized properties. Ugh.
// int g3[ sizeof(I1) == 0 ? 1 : -1];

// rdar://6821047
int bar(I0 *P) {
  P = P+5;  // expected-error {{arithmetic on pointer to interface 'I0', which is not a constant size for this architecture and platform}}
  P = 5+P;  // expected-error {{arithmetic on pointer to interface 'I0', which is not a constant size for this architecture and platform}}
  P = P-5;  // expected-error {{arithmetic on pointer to interface 'I0', which is not a constant size for this architecture and platform}}
  
  return P[4].x[2];  // expected-error {{expected method to read array element not found on object of type 'I0 *'}}
}


@interface I @end

@interface XCAttributeRunDirectNode
{
    @public
    unsigned long attributeRuns[1024 + sizeof(I)]; // expected-error {{application of 'sizeof' to interface 'I' is not supported on this architecture and platform}}
    int i;
}
@end

@implementation XCAttributeRunDirectNode

- (unsigned long)gatherStats:(id )stats
{
        return attributeRuns[i];
}
@end


@interface Foo @end

int foo()
{
  Foo *f;
  
  // Both of these crash clang nicely
  ++f; 	// expected-error {{arithmetic on pointer to interface 'Foo', which is not a constant size for this architecture and platform}}
  --f; 	// expected-error {{arithmetic on pointer to interface 'Foo', which is not a constant size for this architecture and platform}}
}
