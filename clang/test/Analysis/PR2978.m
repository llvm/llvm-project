// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core,osx.cocoa.Dealloc %s -verify

// Tests for the checker which checks missing/extra ivar 'release' calls 
// in dealloc.

@interface NSObject
- (void)release;
- (void)dealloc;
@end

@interface MyClass : NSObject {
@private
  id _X;
  id _Y;
  id _Z;
  id _K;
  id _L;
  id _N;
  id _M;
  id _P;
  id _Q;
  id _R;
  id _S;
  id _V;
  id _W;

  MyClass *_other;

  id _nonPropertyIvar;
}
@property(retain) id X;
@property(retain) id Y;
@property(assign) id Z;
@property(assign) id K;
@property(weak) id L;
@property(readonly) id N;
@property(retain) id M;
@property(weak) id P;
@property(weak) id Q;
@property(retain) id R;
@property(weak, readonly) id S;

@property(assign, readonly) id T; // Shadowed in class extension
@property(assign) id U;

@property(retain) id V;
@property(retain) id W;
-(id) O;
-(void) setO: (id) arg;
@end

@interface MyClass ()
// Shadows T to make it readwrite internally but readonly externally.
@property(assign, readwrite) id T;
@end

@implementation MyClass
@synthesize X = _X;
@synthesize Y = _Y;
@synthesize Z = _Z;
@synthesize K = _K;
@synthesize L = _L;
@synthesize N = _N;
@synthesize M = _M;
@synthesize Q = _Q;
@synthesize R = _R;
@synthesize V = _V;
@synthesize W = _W;

-(id) O{ return 0; }
-(void) setO:(id)arg { }


-(void) releaseInHelper {
  [_R release]; // no-warning
  _R = @"Hi";
}

- (void)dealloc
{

  [_X release];
  [_Z release]; // expected-warning{{The '_Z' ivar in 'MyClass' was synthesized for an assign, readwrite property but was released in 'dealloc'}}
  [_T release]; // no-warning

  [_other->_Z release]; // no-warning
  [_N release];

  self.M = 0; // This will release '_M'
  [self setV:0]; // This will release '_V'
  [self setW:@"newW"]; // This will release '_W', but retain the new value

  [_S release]; // expected-warning {{The '_S' ivar in 'MyClass' was synthesized for a weak property but was released in 'dealloc'}}

  self.O = 0; // no-warning

  [_Q release]; // expected-warning {{The '_Q' ivar in 'MyClass' was synthesized for a weak property but was released in 'dealloc'}}

  self.P = 0;

  [self releaseInHelper];

  [_nonPropertyIvar release]; // no-warning

  // Silly, but not an error.
  if (!_U)
    [_U release];

  [super dealloc];
  // expected-warning@-1{{The '_Y' ivar in 'MyClass' was retained by a synthesized property but not released before '[super dealloc]'}}
  // expected-warning@-2{{The '_W' ivar in 'MyClass' was retained by a synthesized property but not released before '[super dealloc]'}}

}

@end

