// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface I0 
@property(readonly) int x;
@property(readonly) int y;
@property(readonly) int z;
-(void) setY: (int) y0;
@end

@interface I0 (Cat0)
-(void) setX: (int) a0;
@end

@implementation I0
@dynamic x;
@dynamic y;
@dynamic z;
-(void) setY: (int) y0{}

-(void) im0 {
  self.x = 0;
  self.y = 2;
  self.z = 2; // expected-error {{assignment to readonly property}}
}
@end

// Test when property is 'readonly' but it has a setter in
// its implementation only.
@interface I1  {
}
@property(readonly) int identifier;
@end


@implementation I1
@dynamic identifier;
- (void)setIdentifier:(int)ident {}

- (id)initWithIdentifier:(int)Arg {
    self.identifier = 0;
}

@end


// Also in a category implementation
@interface I1(CAT)  
@property(readonly) int rprop;
@end


@implementation I1(CAT)
@dynamic rprop;
- (void)setRprop:(int)ident {}

- (id)initWithIdentifier:(int)Arg {
    self.rprop = 0;
}

@end

static int g_val;

@interface Root 
+ alloc;
- init;
@end

@interface Subclass : Root
{
    int setterOnly;
}
- (void) setSetterOnly:(int)value;
@end

@implementation Subclass
- (void) setSetterOnly:(int)value {
    setterOnly = value;
    g_val = setterOnly;
}
@end

@interface C {}
// - (int)Foo;
- (void)setFoo:(int)value;
@end

void g(int); // expected-note {{passing argument to parameter here}}

void f(C *c) {
    c.Foo = 17; // OK 
    g(c.Foo); // expected-error {{expected getter method not found on object of type 'C *'}}
}


void abort(void);
int main (void) {
    Subclass *x = [[Subclass alloc] init];

    x.setterOnly = 4;   // OK
    if (g_val != 4)
      abort ();
    return 0;
}

// rdar://11363363
@interface rdar11363363
{
  id R;
}
@property (copy) id p;
@property (copy) id r;
@property (copy) id Q;
@property (copy) id t; // expected-note 2 {{property declared here}}
@property (copy) id T; // expected-note 2 {{property declared here}}
@property (copy) id Pxyz; // expected-note 2 {{property declared here}}
@property (copy) id pxyz; // expected-note 2 {{property declared here}}
@end

@implementation rdar11363363
@synthesize p;
@synthesize r;
@synthesize Q;
@synthesize t, T;
@synthesize Pxyz, pxyz;
- (id) Meth {
  self.P = 0;
  self.q = 0;
// rdar://11528439
  self.t = 0; // expected-error {{synthesized properties 't' and 'T' both claim setter 'setT:'}}
  self.T = 0; // expected-error {{synthesized properties 'T' and 't' both claim setter 'setT:'}}
  self.Pxyz = 0; // expected-error {{synthesized properties 'Pxyz' and 'pxyz' both claim setter 'setPxyz:'}}
  self.pxyz = 0; // expected-error {{synthesized properties 'pxyz' and 'Pxyz' both claim setter 'setPxyz:'}}
  self.R = 0;
  return self.R; // expected-error {{expected getter method not found on object of type 'rdar11363363 *'}}
}
@end

// rdar://11499742
@class BridgeFormatter;

@interface FMXBridgeFormatter 

@property(assign, readwrite, getter=formatter, setter=setFormatter:) BridgeFormatter* cppFormatter;

@end

@implementation FMXBridgeFormatter
@synthesize cppFormatter;

- (void) dealloc
{
 self.formatter = 0;
}
@end

