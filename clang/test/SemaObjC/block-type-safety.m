// RUN: %clang_cc1 -fsyntax-only -verify -fblocks -Wno-objc-root-class %s
// RUN: %clang_cc1 -fsyntax-only -verify -fblocks -Wno-objc-root-class \
// RUN:   -fcompatibility-qualified-id-block-type-checking -DCOMPATIBILITY_QUALIFIED_ID_TYPE_CHECKING=1 %s
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
@class NSObject;

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

#ifndef COMPATIBILITY_QUALIFIED_ID_TYPE_CHECKING
int test5() {
    // Returned value is used outside of a block, so error on changing
    // a return type to a more general than expected.
    NSAllArray *(^block)(id);
    id <Foo> (^genericBlock)(id);
    genericBlock = block;
    block = genericBlock; // expected-error {{incompatible block pointer types assigning to 'NSAllArray *(^)(id)' from 'id<Foo> (^)(id)'}}

    // A parameter is used inside a block, so error on changing a parameter type
    // to a more specific than an argument type it will be called with.
    // rdar://problem/52788423
    void (^blockWithParam)(NSAllArray *);
    void (^genericBlockWithParam)(id<Foo>);
    genericBlockWithParam = blockWithParam; // expected-error {{incompatible block pointer types assigning to 'void (^)(id<Foo>)' from 'void (^)(NSAllArray *)'}}
    blockWithParam = genericBlockWithParam;
    return 0;
}
#else
// In Apple SDK APIs using NSItemProviderCompletionHandler require to work with
// blocks that have parameters more specific than in method signatures. As
// explained in non-compatibility test above, it is not safe in general. But
// to keep existing code working we support a compatibility mode that uses
// previous type checking.
int test5() {
    NSAllArray *(^block)(id);
    id <Foo> (^genericBlock)(id);
    genericBlock = block;
    block = genericBlock; // expected-error {{incompatible block pointer types assigning to 'NSAllArray *(^)(id)' from 'id<Foo> (^)(id)'}}

    void (^blockWithParam)(NSAllArray *);
    void (^genericBlockWithParam)(id<Foo>);
    genericBlockWithParam = blockWithParam;
    blockWithParam = genericBlockWithParam;
    return 0;
}
#endif

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

// rdar://16739120
@protocol P1 @end
@protocol P2 @end

void Test() {
void (^aBlock)();
id anId = aBlock;  // OK

id<P1,P2> anQualId = aBlock;  // expected-error {{initializing 'id<P1,P2>' with an expression of incompatible type 'void (^)()'}}

NSArray* anArray = aBlock; // expected-error {{initializing 'NSArray *' with an expression of incompatible type 'void (^)()'}}

aBlock = anId; // OK

id<P1,P2> anQualId1;
aBlock = anQualId1; // expected-error {{assigning to 'void (^)()' from incompatible type 'id<P1,P2>'}}

NSArray* anArray1;
aBlock = anArray1; // expected-error {{assigning to 'void (^)()' from incompatible type 'NSArray *'}}
}

void Test2() {
  void (^aBlock)();
  id<NSObject> anQualId1 = aBlock; // Ok
  id<NSObject, NSCopying> anQualId2 = aBlock; // Ok
  id<NSObject, NSCopying, NSObject, NSCopying> anQualId3 = aBlock; // Ok
  id <P1>  anQualId4  = aBlock; // expected-error {{initializing 'id<P1>' with an expression of incompatible type 'void (^)()'}}
  id<NSObject, P1, NSCopying> anQualId5 = aBlock; // expected-error {{initializing 'id<NSObject,P1,NSCopying>' with an expression of incompatible type 'void (^)()'}}
  id<NSCopying> anQualId6 = aBlock; // Ok
}

void Test3() {
  void (^aBlock)();
  NSObject *NSO = aBlock; // Ok
  NSObject<NSObject> *NSO1 = aBlock; // Ok
  NSObject<NSObject, NSCopying> *NSO2 = aBlock; // Ok
  NSObject<NSObject, NSCopying, NSObject, NSCopying> *NSO3 = aBlock; // Ok
  NSObject <P1>  *NSO4  = aBlock; // expected-error {{initializing 'NSObject<P1> *' with an expression of incompatible type 'void (^)()'}}
  NSObject<NSObject, P1, NSCopying> *NSO5 = aBlock; // expected-error {{initializing 'NSObject<NSObject,P1,NSCopying> *' with an expression of incompatible type 'void (^)()'}}
  NSObject<NSCopying> *NSO6 = aBlock; // Ok
}

// rdar://problem/19420731
typedef NSObject<P1> NSObject_P1;
typedef NSObject_P1<P2> NSObject_P1_P2;

void Test4(void (^handler)(NSObject_P1_P2 *p)) {
  Test4(^(NSObject<P2> *p) { });
}
