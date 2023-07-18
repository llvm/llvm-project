// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef struct { int y; } Abstract;

typedef struct { int x; } Alternate;

#define INTERFERE_TYPE Alternate*

@protocol A
@property Abstract *x; // expected-note {{using}}
@end

@interface B
@property Abstract *y; // expected-note {{using}}
@end

@interface B (Category)
@property Abstract *z; // expected-note {{using}}
@end

@interface InterferencePre
-(void) x; // expected-note {{also found}}
-(void) y; // expected-note {{also found}}
-(void) z; // expected-note {{also found}}
-(void) setX: (INTERFERE_TYPE) arg; 
-(void) setY: (INTERFERE_TYPE) arg;
-(void) setZ: (INTERFERE_TYPE) arg;
@end

void f0(id a0) {
  Abstract *l = [a0 x]; // expected-warning {{multiple methods named 'x' found}} 
}

void f1(id a0) {
  Abstract *l = [a0 y]; // expected-warning {{multiple methods named 'y' found}}
}

void f2(id a0) {
  Abstract *l = [a0 z]; // expected-warning {{multiple methods named 'z' found}}
}

void f3(id a0, Abstract *a1) { 
  [ a0 setX: a1];
}

void f4(id a0, Abstract *a1) { 
  [ a0 setY: a1];
}

void f5(id a0, Abstract *a1) { 
  [ a0 setZ: a1];
}

// pr7861
void f6(id<A> a0) {
  Abstract *l = [a0 x];
}

struct test3a { int x, y; };
struct test3b { unsigned x, y; };
@interface Test3A - (struct test3a) test3; @end
@interface Test3B - (struct test3b) test3; @end
void test3(id x) {
  (void) [x test3];
}

struct test4a { int x, y; };
struct test4b { float x, y; };
@interface Test4A - (struct test4a) test4; @end //expected-note{{using}}
@interface Test4B - (struct test4b) test4; @end //expected-note{{also found}}
void test4(id x) {
  (void) [x test4]; //expected-warning {{multiple methods named 'test4' found}}
}

// rdar://19265296
#pragma clang diagnostic ignored "-Wobjc-multiple-method-names"
@interface NSObject 
+ (id)alloc;
+ (id)class;
- (id) init;
@end

@class NSString;
@interface A : NSObject
- (instancetype)initWithType:(NSString *)whatever;
@end

@interface Test : NSObject @end

@implementation Test
+ (instancetype)foo
{
    return [[[self class] alloc] initWithType:3];
}
- (instancetype)initWithType:(int)whatever
{
    return [super init];
}
@end
