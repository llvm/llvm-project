// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

typedef struct objc_object {
  Class isa;
} *id;


@interface foo
- (void)meth;
@end

@implementation foo
- (void) contents {}			// No declaration in @interface!
- (void) meth { [self contents]; } 
@end

typedef struct _NSPoint {
    float x;
    float y;
} NSPoint;

typedef struct _NSSize {
    float width; 
    float height;
} NSSize;

typedef struct _NSRect {
    NSPoint origin;
    NSSize size;
} NSRect;

@interface AnyClass
- (NSRect)rect;
@end

@class Helicopter;

static void func(Helicopter *obj) {
  // Note that the proto for "rect" is found in the global pool even when
  // a statically typed object's class interface isn't in scope! This 
  // behavior isn't very desirable, however wee need it for GCC compatibility.
  NSRect r = [obj rect];
}

@interface NSObject @end

extern Class NSClassFromObject(id object);

@interface XX : NSObject 
@end

@implementation XX

+ _privateMethod {
  return self;
}

- (void) xx {
  [NSClassFromObject(self) _privateMethod];
}
@end

@implementation XX (Private)
- (void) yy {
  [NSClassFromObject(self) _privateMethod];
}
@end

@interface I0
-(void) nonVararg: (int) x;
@end

int f0(I0 *ob) {
  [ ob nonVararg: 0, 1, 2]; // expected-error {{too many arguments to method call}}
}

int f2(void) {
    const id foo;
    [foo bar];  // expected-warning {{method '-bar' not found (return type defaults to 'id')}}
    return 0;
}


// PR3766
struct S { int X; } S;

int test5(int X) {
  int a = [X somemsg];  // expected-warning {{receiver type 'int' is not 'id'}} \
                           expected-warning {{method '-somemsg' not found}} \
                           expected-error {{incompatible pointer to integer conversion initializing 'int' with an expression of type 'id'}}
  int b = [S somemsg];  // expected-error {{bad receiver type 'struct S'}}
}

// PR4021
void foo4(void) {
  struct objc_object X[10];
  
  [X rect]; // expected-warning {{receiver type 'struct objc_object *' is not 'id' or interface pointer, consider casting it to 'id'}}
}

void foo5(id p) {
  p
  [(id)(p) bar]; // expected-error {{missing '['}} \
                 // expected-error {{expected ']'}} \
                 // expected-note {{to match this '['}} \
                 // expected-warning {{instance method '-bar' not found}}
}

@interface I1 // expected-note {{receiver is instance of class declared here}}
-(void)unavail_meth  __attribute__((unavailable)); // expected-note {{marked unavailable here}}
@end

void foo6(I1 *p) {
  [p
    bar]; // expected-warning {{instance method '-bar' not found}}
  [p
    unavail_meth]; // expected-error {{unavailable}}
}
