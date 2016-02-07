// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.core -analyzer-checker=alpha.osx.cocoa.Dealloc %s -verify

// Tests for the checker which checks missing/extra ivar 'release' calls 
// in dealloc.

@interface NSObject
- (void)release;
- dealloc;
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
  id _V;
  id _W;
}
@property(retain) id X;
@property(retain) id Y;
@property(assign) id Z;
@property(assign) id K;
@property(weak) id L;
@property(readonly) id N;
@property(retain) id M;
@property(retain) id V;
@property(retain) id W;
-(id) O;
-(void) setO: (id) arg;
@end

@implementation MyClass
@synthesize X = _X;
@synthesize Y = _Y; // expected-warning{{The '_Y' instance variable in 'MyClass' was retained by a synthesized property but was not released in 'dealloc'}}
@synthesize Z = _Z; // expected-warning{{The '_Z' instance variable in 'MyClass' was not retained by a synthesized property but was released in 'dealloc'}}
@synthesize K = _K;
@synthesize L = _L; // no-warning
@synthesize N = _N; // no-warning
@synthesize M = _M;
@synthesize V = _V;
@synthesize W = _W; // expected-warning{{The '_W' instance variable in 'MyClass' was retained by a synthesized property but was not released in 'dealloc'}}

-(id) O{ return 0; }
-(void) setO:(id)arg { }

- (id)dealloc
{
  [_X release];
  [_Z release];
  [_N release];
  
  self.M = 0; // This will release '_M'
  [self setV:0]; // This will release '_V'
  [self setW:@"newW"]; // This will release '_W', but retain the new value
  self.O = 0; // no-warning  
  [super dealloc];
  return 0;
}

@end

