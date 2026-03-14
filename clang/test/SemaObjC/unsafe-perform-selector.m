// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -verify %s

@class Thread;

__attribute__((objc_root_class))
@interface NSObject

- (id)performSelector:(SEL)sel;
- (void)performSelectorInBackground:(SEL)sel withObject:(id)arg;
- (void)performSelectorOnMainThread:(SEL)sel;

- (void)performSelectorOnMainThread:(SEL)aSelector
                           onThread:(Thread *)thread
                         withObject:(id)arg
                      waitUntilDone:(int)wait
                              modes:(id *)array;

@end

typedef struct { int x; int y; int width; int height; } Rectangle;

struct Struct { Rectangle r; };

typedef union { int x; float f; } Union;

@interface Base : NSObject

- (struct Struct)returnsStruct2; // expected-note {{method 'returnsStruct2' that returns 'struct Struct' declared here}}
- (Union)returnsId;

@end

@protocol IP

- (Union)returnsUnion; // expected-note 2 {{method 'returnsUnion' that returns 'Union' declared here}}

@end

typedef __attribute__((__ext_vector_type__(3))) float float3;
typedef int int4 __attribute__ ((vector_size (16)));

@interface I : Base<IP>

- (Rectangle)returnsStruct; // expected-note 4 {{method 'returnsStruct' that returns 'Rectangle' declared here}}
- (id)returnsId; // shadows base 'returnsId'
- (int)returnsInt;
- (I *)returnPtr;
- (float3)returnsExtVector; // expected-note {{method 'returnsExtVector' that returns 'float3' (vector of 3 'float' values) declared here}}
- (int4)returnsVector; // expected-note {{method 'returnsVector' that returns 'int4' (vector of 4 'int' values) declared here}}

+ (Rectangle)returnsStructClass; // expected-note 2 {{method 'returnsStructClass' that returns 'Rectangle' declared here}}
+ (void)returnsUnion; // Not really

@end

void foo(I *i) {
  [i performSelector: @selector(returnsStruct)]; // expected-warning {{'performSelector:' is incompatible with selectors that return a struct type}}
  [i performSelectorInBackground: @selector(returnsStruct) withObject:0]; // expected-warning {{'performSelectorInBackground:withObject:' is incompatible with selectors that return a struct type}}
  [i performSelector: ((@selector(returnsUnion)))]; // expected-warning {{'performSelector:' is incompatible with selectors that return a union type}}
  [i performSelectorOnMainThread: @selector(returnsStruct2)]; // expected-warning {{'performSelectorOnMainThread:' is incompatible with selectors that return a struct type}}
  [I performSelector: (@selector(returnsStructClass))]; // expected-warning {{'performSelector:' is incompatible with selectors that return a struct type}}

  [i performSelector: @selector(returnsId)];
  [i performSelector: @selector(returnsInt)];
  [i performSelector: @selector(returnsPtr)];
  [I performSelector: @selector(returnsUnion)]; // No warning expected

  id obj = i;
  [obj performSelector: @selector(returnsId)];
  [obj performSelector: @selector(returnsStruct)];
}

@interface SubClass: I

@end

@interface SubClass ()
- (struct Struct)returnsSubStructExt; // expected-note {{method 'returnsSubStructExt' that returns 'struct Struct' declared here}} expected-note {{method 'returnsSubStructExt' declared here}}
@end

@implementation SubClass // expected-warning {{method definition for 'returnsSubStructExt' not found}}

- (struct Struct)returnsSubStructImpl { // expected-note {{method 'returnsSubStructImpl' that returns 'struct Struct' declared here}}
  struct Struct Result;
  return Result;
}

- (void)checkPrivateCalls {
  [self performSelector: @selector(returnsSubStructExt)]; // expected-warning {{'performSelector:' is incompatible with selectors that return a struct type}}
  [self performSelector: @selector(returnsSubStructImpl)]; // expected-warning {{'performSelector:' is incompatible with selectors that return a struct type}}
}

- (void)checkSuperCalls {
  [super performSelector: @selector(returnsStruct)]; // expected-warning {{'performSelector:' is incompatible with selectors that return a struct type}}
  [super performSelectorInBackground: @selector(returnsUnion) withObject: self]; // expected-warning {{'performSelectorInBackground:withObject:' is incompatible with selectors that return a union type}}
  [super performSelector: @selector(returnsId)];
}

+ (struct Struct)returnsSubStructClassImpl { // expected-note {{method 'returnsSubStructClassImpl' that returns 'struct Struct' declared here}}
  struct Struct Result;
  return Result;
}

+ (void)checkClassPrivateCalls {
  [self performSelector: @selector(returnsSubStructClassImpl)]; // expected-warning {{'performSelector:' is incompatible with selectors that return a struct type}}
}

+ (void)checkClassSuperCalls {
  [super performSelector: @selector(returnsStructClass)]; // expected-warning {{'performSelector:' is incompatible with selectors that return a struct type}}
  [super performSelector: @selector(returnsUnion)]; // No warning expected
}

@end

@implementation I (LongPerformSelectors)

- (void)checkLongCallsFromCategory {
  [self performSelectorOnMainThread: @selector(returnsStruct) onThread:0 withObject:self waitUntilDone:1 modes:0]; // expected-warning {{'performSelectorOnMainThread:onThread:withObject:waitUntilDone:modes:' is incompatible with selectors that return a struct type}}
}

- (void)checkVectorReturn {
  [self performSelector: @selector(returnsExtVector)]; // expected-warning {{'performSelector:' is incompatible with selectors that return a vector type}}
  [self performSelector: @selector(returnsVector)]; // expected-warning {{'performSelector:' is incompatible with selectors that return a vector type}}
}

@end
