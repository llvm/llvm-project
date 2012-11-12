// RUN: %clang_cc1 -fsyntax-only -verify -fblocks -Wno-objc-root-class %s
// test for block type safety.

@interface Super  @end
@interface Sub : Super @end

void f2(void(^f)(Super *)) { // expected-note{{passing argument to parameter 'f' here}}
    Super *o;
    f(o);
}

void f3(void(^f)(Sub *)) {
    Sub *o;
    f(o);
}

void r0(Super* (^f)()) {
     Super *o = f();
}

void r1(Sub* (^f)()) { // expected-note{{passing argument to parameter 'f' here}}
    Sub *o = f();
}

@protocol NSObject;

void r2 (id<NSObject> (^f) (void)) {
  id o = f();
}

void test1() {
    f2(^(Sub *o) { });    // expected-error {{incompatible block pointer types passing}}
    f3(^(Super *o) { });  // OK, block taking Super* may be called with a Sub*

    r0(^Super* () { return 0; });  // OK
    r0(^Sub* () { return 0; });    // OK, variable of type Super* gets return value of type Sub*
    r0(^id () { return 0; });

    r1(^Super* () { return 0; });  // expected-error {{incompatible block pointer types passing}}
    r1(^Sub* () { return 0; });    // OK
    r1(^id () { return 0; }); 
     
    r2(^id<NSObject>() { return 0; });
}


@interface A @end
@interface B @end

void f0(void (^f)(A* x)) {
  A* a;
  f(a);
}

void f1(void (^f)(id x)) {
  B *b;
  f(b);
}

void test2(void) 
{ 
  f0(^(id a) { }); // OK
  f1(^(A* a) { });
   f1(^(id<NSObject> a) { });	// OK
}

@interface NSArray
   // Calls block() with every object in the array
   -enumerateObjectsWithBlock:(void (^)(id obj))block;
@end

@interface MyThing
-(void) printThing;
@end

@implementation MyThing
    static NSArray* myThings;  // array of MyThing*

   -(void) printThing {  }

// programmer wants to write this:
   -printMyThings1 {
       [myThings enumerateObjectsWithBlock: ^(MyThing *obj) {
           [obj printThing];
       }];
   }

// strict type safety requires this:
   -printMyThings {
       [myThings enumerateObjectsWithBlock: ^(id obj) {
           MyThing *obj2 = (MyThing *)obj;
           [obj2 printThing];
       }];
   }
@end

@protocol P, P2;
void f4(void (^f)(id<P> x)) { // expected-note{{passing argument to parameter 'f' here}}
    NSArray<P2> *b;
    f(b);	// expected-warning {{passing 'NSArray<P2> *' to parameter of incompatible type 'id<P>'}}
}

void test3() {
  f4(^(NSArray<P2>* a) { });  // expected-error {{incompatible block pointer types passing 'void (^)(NSArray<P2> *)' to parameter of type 'void (^)(id<P>)'}}
}

// rdar : //8302845
@protocol Foo @end

@interface Baz @end

@interface Baz(FooConformance) <Foo>
@end

@implementation Baz @end

int test4 () {
    id <Foo> (^b)() = ^{ // Doesn't work
        return (Baz *)0;
    };
    return 0;
}

// rdar:// 9118343

@protocol NSCopying @end

@interface NSAllArray <NSCopying>
@end

@interface NSAllArray (FooConformance) <Foo>
@end

int test5() {
    NSAllArray *(^block)(id);
    id <Foo> (^genericBlock)(id);
    genericBlock = block;
    return 0;
}

// rdar://10798770
typedef int NSInteger;

typedef enum : NSInteger {NSOrderedAscending = -1L, NSOrderedSame, NSOrderedDescending} NSComparisonResult;

typedef NSComparisonResult (^NSComparator)(id obj1, id obj2);

@interface radar10798770
- (void)sortUsingComparator:(NSComparator)c;
@end

void f() {
   radar10798770 *f;
   [f sortUsingComparator:^(id a, id b) {
        return NSOrderedSame;
   }];
}
