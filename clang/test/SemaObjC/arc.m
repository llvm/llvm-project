// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fobjc-runtime-has-weak -fsyntax-only -fobjc-arc -fblocks -verify -Wno-objc-root-class %s

typedef unsigned long NSUInteger;
typedef const void * CFTypeRef;
CFTypeRef CFBridgingRetain(id X);
id CFBridgingRelease(CFTypeRef);
@protocol NSCopying @end
@interface NSDictionary
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id <NSCopying> [])keys count:(NSUInteger)cnt;
- (void)setObject:(id)object forKeyedSubscript:(id)key;
@end
@class NSFastEnumerationState;
@protocol NSFastEnumeration
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id __unsafe_unretained [])buffer count:(NSUInteger)len;
@end
@interface NSNumber 
+ (NSNumber *)numberWithInt:(int)value;
@end
@interface NSArray <NSFastEnumeration>
+ (id)arrayWithObjects:(const id [])objects count:(NSUInteger)cnt;
@end

void test0(void (*fn)(int), int val) {
  fn(val);
}

@interface A
- (id)retain;
- (id)autorelease;
- (oneway void)release;
- (void)dealloc;
- (NSUInteger)retainCount;
@end

void test1(A *a) {
  SEL s = @selector(retain);	// expected-error {{ARC forbids use of 'retain' in a @selector}}
  s = @selector(release);	// expected-error {{ARC forbids use of 'release' in a @selector}}
  s = @selector(autorelease);	// expected-error {{ARC forbids use of 'autorelease' in a @selector}}
  s = @selector(dealloc);	// expected-error {{ARC forbids use of 'dealloc' in a @selector}}
  [a dealloc]; // expected-error {{ARC forbids explicit message send of 'dealloc'}}
  [a retain]; // expected-error {{ARC forbids explicit message send of 'retain'}}
  [a retainCount]; // expected-error {{ARC forbids explicit message send of 'retainCount'}}
  [a release]; // expected-error {{ARC forbids explicit message send of 'release'}}
  [a autorelease]; // expected-error {{ARC forbids explicit message send of 'autorelease'}}
}

@interface Test2 : A
- (void) dealloc;
@end
@implementation Test2
- (void) dealloc {
  // This should maybe just be ignored.  We're just going to warn about it for now.
  [super dealloc]; // expected-error {{ARC forbids explicit message send of 'dealloc'}}
}
@end

// rdar://8843638

@interface I
- (id)retain; // expected-note {{method 'retain' declared here}}
- (id)autorelease; // expected-note {{method 'autorelease' declared here}}
- (oneway void)release; // expected-note {{method 'release' declared here}}
- (NSUInteger)retainCount; // expected-note {{method 'retainCount' declared here}}
@end

@implementation I
- (id)retain{return 0;} // expected-error {{ARC forbids implementation of 'retain'}}
- (id)autorelease{return 0;} // expected-error {{ARC forbids implementation of 'autorelease'}}
- (oneway void)release{} // expected-error {{ARC forbids implementation of 'release'}}
- (NSUInteger)retainCount{ return 0; } // expected-error {{ARC forbids implementation of 'retainCount'}}
@end

@implementation I(CAT)
- (id)retain{return 0;} // expected-error {{ARC forbids implementation of 'retain'}} \
                        // expected-warning {{category is implementing a method which will also be implemented by its primary class}}
- (id)autorelease{return 0;} // expected-error {{ARC forbids implementation of 'autorelease'}} \
                         // expected-warning {{category is implementing a method which will also be implemented by its primary class}}
- (oneway void)release{} // expected-error {{ARC forbids implementation of 'release'}} \
                          // expected-warning {{category is implementing a method which will also be implemented by its primary class}}
- (NSUInteger)retainCount{ return 0; } // expected-error {{ARC forbids implementation of 'retainCount'}} \
                          // expected-warning {{category is implementing a method which will also be implemented by its primary class}}
@end

// rdar://8861761

@interface B
+ (id)alloc;
- (id)initWithInt: (int) i;
- (id)myInit __attribute__((objc_method_family(init)));
- (id)myBadInit __attribute__((objc_method_family(12)));  // expected-error {{'objc_method_family' attribute requires parameter 1 to be an identifier}}

@end

void rdar8861761() {
  B *o1 = [[B alloc] initWithInt:0];
  B *o2 = [B alloc];
  [o2 initWithInt:0]; // expected-warning {{expression result unused}}
  B *o3 = [[B alloc] myInit];
  [[B alloc] myInit]; // expected-warning {{expression result unused}}
}

// rdar://8925835
@interface rdar8925835
- (void)foo:(void (^)(unsigned captureCount, I * const capturedStrings[captureCount]))block;
@end

void test5() {
  extern void test5_helper(__autoreleasing id *);
  id x;

  // Okay because of magic temporaries.
  test5_helper(&x);

  __autoreleasing id *a = &x; // expected-error {{initializing '__autoreleasing id *' with an expression of type '__strong id *' changes retain/release properties of pointer}}

  a = &x; // expected-error {{assigning '__strong id *' to '__autoreleasing id *' changes retain/release properties of pointer}}

  extern void test5_helper2(id const *);
  test5_helper2(&x);

  extern void test5_helper3(__weak id *); // expected-note {{passing argument to parameter here}}
  test5_helper3(&x); // expected-error {{passing '__strong id *' to parameter of type '__weak id *' changes retain/release properties of pointer}}
}

// rdar://problem/8937869
void test6(unsigned cond) {
  switch (cond) {
  case 0:
    ;
    id x; // expected-note {{jump bypasses initialization of __strong variable}}

  case 1: // expected-error {{cannot jump}}
    break;
  }
}
void test6a(unsigned cond) {
  switch (cond) {
  case 0:
    ;
    __weak id x; // expected-note {{jump bypasses initialization of __weak variable}}

  case 1: // expected-error {{cannot jump}}
    break;
  }
}

@class NSError;
void test7(void) {
  extern void test7_helper(NSError **);
  NSError *err;
  test7_helper(&err);
}
void test7_weak(void) {
  extern void test7_helper(NSError **);
  __weak NSError *err;
  test7_helper(&err);
}
void test7_unsafe(void) {
  extern void test7_helper(NSError **); // expected-note {{passing argument to parameter here}}
  __unsafe_unretained NSError *err;
  test7_helper(&err); // expected-error {{passing 'NSError *__unsafe_unretained *' to parameter of type 'NSError *__autoreleasing *' changes retain/release properties of pointer}}
}

@class Test8_incomplete;
@interface Test8_complete @end;
@interface Test8_super @end;
@interface Test8 : Test8_super
- (id) init00;
- (id) init01; // expected-note {{declaration in interface}} \
               // expected-note{{overridden method}}
- (id) init02; // expected-note{{overridden method}}
- (id) init03; // covariance
- (id) init04; // covariance
- (id) init05; // expected-note{{overridden method}}

- (void) init10; // expected-note {{declaration in interface is not in the 'init' family because its result type is not an object pointer}}
- (void) init11;
- (void) init12;
- (void) init13; // expected-note {{declaration in interface is not in the 'init' family because its result type is not an object pointer}}
- (void) init14; // expected-note {{declaration in interface is not in the 'init' family because its result type is not an object pointer}}
- (void) init15;

// These should be invalid to actually call.
- (Test8_incomplete*) init20;
- (Test8_incomplete*) init21; // expected-note {{declaration in interface}}
- (Test8_incomplete*) init22;
- (Test8_incomplete*) init23;
- (Test8_incomplete*) init24;
- (Test8_incomplete*) init25;

- (Test8_super*) init30; // id exception to covariance
- (Test8_super*) init31; // expected-note {{declaration in interface}} \
                         // expected-note{{overridden method}}
- (Test8_super*) init32; // expected-note{{overridden method}}
- (Test8_super*) init33;
- (Test8_super*) init34; // covariance
- (Test8_super*) init35; // expected-note{{overridden method}}

- (Test8*) init40; // id exception to covariance
- (Test8*) init41; // expected-note {{declaration in interface}} \
                   // expected-note{{overridden method}}
- (Test8*) init42; // expected-note{{overridden method}}
- (Test8*) init43; // this should be a warning, but that's a general language thing, not an ARC thing
- (Test8*) init44;
- (Test8*) init45; // expected-note{{overridden method}}

- (Test8_complete*) init50; // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_complete*) init51; // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_complete*) init52; // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_complete*) init53; // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_complete*) init54; // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_complete*) init55; // expected-error {{init methods must return a type related to the receiver type}}
@end
@implementation Test8
- (id) init00 { return 0; }
- (id) init10 { return 0; } // expected-error {{method implementation does not match its declaration}}
- (id) init20 { return 0; }
- (id) init30 { return 0; }
- (id) init40 { return 0; }
- (id) init50 { return 0; }

- (void) init01 {} // expected-error {{method was declared as an 'init' method, but its implementation doesn't match because its result type is not an object pointer}} \
                   // expected-warning{{method is expected to return an instance of its class type 'Test8', but is declared to return 'void'}}
- (void) init11 {}
- (void) init21 {} // expected-error {{method was declared as an 'init' method, but its implementation doesn't match because its result type is not an object pointer}}
- (void) init31 {} // expected-error {{method was declared as an 'init' method, but its implementation doesn't match because its result type is not an object pointer}} \
                   // expected-warning{{method is expected to return an instance of its class type 'Test8', but is declared to return 'void'}}
- (void) init41 {} // expected-error {{method was declared as an 'init' method, but its implementation doesn't match because its result type is not an object pointer}} \
                   // expected-warning{{method is expected to return an instance of its class type 'Test8', but is declared to return 'void'}}
- (void) init51 {}

- (Test8_incomplete*) init02 { return 0; } // expected-error {{init methods must return a type related to the receiver type}} \
                                           // expected-warning{{method is expected to return an instance of its class type 'Test8', but is declared to return 'Test8_incomplete *'}}
- (Test8_incomplete*) init12 { return 0; } // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_incomplete*) init22 { return 0; } // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_incomplete*) init32 { return 0; } // expected-error {{init methods must return a type related to the receiver type}} \
                                           // expected-warning{{method is expected to return an instance of its class type 'Test8', but is declared to return 'Test8_incomplete *'}}
- (Test8_incomplete*) init42 { return 0; } // expected-error {{init methods must return a type related to the receiver type}} \
                                           // expected-warning{{method is expected to return an instance of its class type 'Test8', but is declared to return 'Test8_incomplete *'}}
- (Test8_incomplete*) init52 { return 0; } // expected-error {{init methods must return a type related to the receiver type}}

- (Test8_super*) init03 { return 0; }
- (Test8_super*) init13 { return 0; } // expected-error {{method implementation does not match its declaration}}
- (Test8_super*) init23 { return 0; }
- (Test8_super*) init33 { return 0; }
- (Test8_super*) init43 { return 0; }
- (Test8_super*) init53 { return 0; }

- (Test8*) init04 { return 0; }
- (Test8*) init14 { return 0; } // expected-error {{method implementation does not match its declaration}}
- (Test8*) init24 { return 0; }
- (Test8*) init34 { return 0; }
- (Test8*) init44 { return 0; }
- (Test8*) init54 { return 0; }

- (Test8_complete*) init05 { return 0; } // expected-error {{init methods must return a type related to the receiver type}} \
                                         // expected-warning{{method is expected to return an instance of its class type 'Test8', but is declared to return 'Test8_complete *'}}
- (Test8_complete*) init15 { return 0; } // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_complete*) init25 { return 0; } // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_complete*) init35 { return 0; } // expected-error {{init methods must return a type related to the receiver type}} \
                                         // expected-warning{{method is expected to return an instance of its class type 'Test8', but is declared to return 'Test8_complete *'}}
- (Test8_complete*) init45 { return 0; } // expected-error {{init methods must return a type related to the receiver type}} \
                                         // expected-warning{{method is expected to return an instance of its class type 'Test8', but is declared to return 'Test8_complete *'}}
- (Test8_complete*) init55 { return 0; } // expected-error {{init methods must return a type related to the receiver type}}
@end

@class Test9_incomplete;
@interface Test9
- (Test9_incomplete*) init1; // expected-error {{init methods must return a type related to the receiver type}}
- (Test9_incomplete*) init2;
@end
id test9(Test9 *v) {
  return [v init1];
}

// Test that the inference rules are different for fast enumeration variables.
void test10(id collection) {
  for (id x in collection) {
    __strong id *ptr = &x; // expected-warning {{initializing '__strong id *' with an expression of type 'const __strong id *' discards qualifiers}}
  }

  for (__strong id x in collection) {
    __weak id *ptr = &x; // expected-error {{initializing '__weak id *' with an expression of type '__strong id *' changes retain/release properties of pointer}}
  }
}

// rdar://problem/9078626
#define nil ((void*) 0)
void test11(id op, void *vp) {
  _Bool b;
  b = (op == nil);
  b = (nil == op);

  b = (vp == nil);
  b = (nil == vp);

  b = (vp == op); // expected-error {{implicit conversion of Objective-C pointer type 'id' to C pointer type 'void *' requires a bridged cast}} expected-note {{use __bridge}} expected-note {{use CFBridgingRetain call}}
  b = (op == vp);
}

void test12(id collection) {
  for (id x in collection) {
    x = 0; // expected-error {{fast enumeration variables cannot be modified in ARC by default; declare the variable __strong to allow this}}
  }

  for (const id x in collection) { // expected-note {{variable 'x' declared const here}}
    x = 0; // expected-error {{cannot assign to variable 'x' with const-qualified type 'const __strong id'}}
  }

  for (__strong id x in collection) {
    x = 0;
  }
}

@interface Test13
- (id) init0;
- (void) noninit;
@end
@implementation Test13
- (id) init0 {
  self = 0;
}
- (void) noninit {
  self = 0; // expected-error {{cannot assign to 'self' outside of a method in the init family}}
}
@end

// <rdar://problem/10274056>
@interface Test13_B
- (id) consumesSelf __attribute__((ns_consumes_self));
@end
@implementation Test13_B
- (id) consumesSelf {
  self = 0; // no-warning
}
@end

// rdar://problem/9172151
@class Test14A, Test14B;
void test14() {
  extern void test14_consume(id *);
  extern int test14_cond(void);
  extern float test14_nowriteback(id __autoreleasing const *); // expected-note{{passing argument to parameter here}}

  Test14A *a;
  Test14B *b;
  id i;
  id cla[10];
  id vla[test14_cond() + 10];

  test14_consume((__strong id*) &a);
  test14_consume((test14_cond() ? (__strong id*) &b : &i));
  test14_consume(test14_cond() ? 0 : &a);
  test14_consume(test14_cond() ? (void*) 0 : (&a));
  test14_consume(cla); // expected-error {{passing address of non-scalar object to __autoreleasing parameter for write-back}}
  test14_consume(vla); // expected-error {{passing address of non-scalar object to __autoreleasing parameter for write-back}}
  test14_consume(&cla[5]); // expected-error {{passing address of non-scalar object to __autoreleasing parameter for write-back}}

  __strong id *test14_indirect(void);
  test14_consume(test14_indirect()); // expected-error {{passing address of non-local object to __autoreleasing parameter for write-back}}

  extern id test14_global;
  test14_consume(&test14_global); // expected-error {{passing address of non-local object to __autoreleasing parameter for write-back}}

  extern __strong id *test14_global_ptr;
  test14_consume(test14_global_ptr); // expected-error {{passing address of non-local object to __autoreleasing parameter for write-back}}

  static id static_local;
  test14_consume(&static_local); // expected-error {{passing address of non-local object to __autoreleasing parameter for write-back}}

  __weak id* wip;
  test14_nowriteback(&static_local); // okay, not a write-back.
  test14_nowriteback(wip); // expected-error{{passing '__weak id *' to parameter of type '__autoreleasing id const *' changes retain/release properties of pointer}}
}

void test15() {
  __block __autoreleasing id x; // expected-error {{__block variables cannot have __autoreleasing ownership}}
}

struct Test16;
@interface Test16a
- (void) test16_0: (int) x;
- (int) test16_1: (int) x; // expected-note {{one possibility}}
- (int) test16_2: (int) x; // expected-note {{one possibility}}
- (id) test16_3: (int) x __attribute__((ns_returns_retained)); // expected-note {{one possibility}}
- (void) test16_4: (int) x __attribute__((ns_consumes_self)); // expected-note {{one possibility}}
- (void) test16_5: (id) __attribute__((ns_consumed)) x; // expected-note {{one possibility}}
- (void) test16_6: (id) x;
@end

@interface Test16b 
- (void) test16_0: (int) x;
- (int) test16_1: (char*) x; // expected-note {{also found}}
- (char*) test16_2: (int) x; // expected-note {{also found}}
- (id) test16_3: (int) x; // expected-note {{also found}}
- (void) test16_4: (int) x; // expected-note {{also found}}
- (void) test16_5: (id) x; // expected-note {{also found}}
- (void) test16_6: (struct Test16 *) x;
@end

void test16(void) {
  id v;
  [v test16_0: 0];
  [v test16_1: 0]; // expected-error {{multiple methods named 'test16_1:' found with mismatched result, parameter type or attributes}}
  [v test16_2: 0]; // expected-error {{multiple methods named}}
  [v test16_3: 0]; // expected-error {{multiple methods named}}
  [v test16_4: 0]; // expected-error {{multiple methods named}}
  [v test16_5: 0]; // expected-error {{multiple methods named}}
  [v test16_6: 0];
}

@class Test17; // expected-note 2{{forward declaration of class here}}
@protocol Test17p
- (void) test17;
+ (void) test17;
@end
void test17(void) {
  Test17 *v0;
  [v0 test17]; // expected-error {{receiver type 'Test17' for instance message is a forward declaration}}

  Test17<Test17p> *v1;
  [v1 test17]; // expected-error {{receiver type 'Test17<Test17p>' for instance message is a forward declaration}}

  [Test17 test17]; // expected-error {{receiver 'Test17' for class message is a forward declaration}}
}

void test18(void) {
  id x;
  [x test18]; // expected-error {{instance method 'test18' not found ; did you mean 'test17'?}}
}

extern struct Test19 *test19a;
struct Test19 *const test19b = 0;
void test19(void) {
  id x;
  x = (id) test19a; // expected-error {{bridged cast}} \
  // expected-note{{use __bridge to convert directly (no change in ownership)}} \
  // expected-note{{use CFBridgingRelease call to transfer ownership of a +1 'struct Test19 *' into ARC}}
  x = (id) test19b; // expected-error {{bridged cast}} \
  // expected-note{{use __bridge to convert directly (no change in ownership)}} \
  // expected-note{{use CFBridgingRelease call to transfer ownership of a +1 'struct Test19 *' into ARC}}
}

// rdar://problem/8951453
static __thread id test20_implicit; // expected-error {{thread-local variable has non-trivial ownership: type is '__strong id'}}
static __thread __strong id test20_strong; // expected-error {{thread-local variable has non-trivial ownership: type is '__strong id'}}
static __thread __weak id test20_weak; // expected-error {{thread-local variable has non-trivial ownership: type is '__weak id'}}
static __thread __autoreleasing id test20_autoreleasing; // expected-error {{thread-local variable has non-trivial ownership: type is '__autoreleasing id'}} expected-error {{global variables cannot have __autoreleasing ownership}}
static __thread __unsafe_unretained id test20_unsafe;
void test20(void) {
  static __thread id test20_implicit; // expected-error {{thread-local variable has non-trivial ownership: type is '__strong id'}}
  static __thread __strong id test20_strong; // expected-error {{thread-local variable has non-trivial ownership: type is '__strong id'}}
  static __thread __weak id test20_weak; // expected-error {{thread-local variable has non-trivial ownership: type is '__weak id'}}
  static __thread __autoreleasing id test20_autoreleasing; // expected-error {{thread-local variable has non-trivial ownership: type is '__autoreleasing id'}} expected-error {{global variables cannot have __autoreleasing ownership}}
  static __thread __unsafe_unretained id test20_unsafe;
}

// rdar://9310049
_Bool fn(id obj) {
    return (_Bool)obj;
}

// Check casting w/ ownership qualifiers.
void test21() {
  __strong id *sip;
  (void)(__weak id *)sip; // expected-error{{casting '__strong id *' to type '__weak id *' changes retain/release properties of pointer}}
  (void)(__weak const id *)sip; // expected-error{{casting '__strong id *' to type '__weak id const *' changes retain/release properties of pointer}}
  (void)(__autoreleasing id *)sip; // expected-error{{casting '__strong id *' to type '__autoreleasing id *' changes retain/release properties of pointer}}
  (void)(__autoreleasing const id *)sip; // okay
}

// rdar://problem/9340462
void test22(id x[]) { // expected-error {{must explicitly describe intended ownership of an object array parameter}}
}

// rdar://problem/9400219
void test23(void) {
  void *ptr;
  ptr = @"foo";
  ptr = (ptr ? @"foo" : 0);
  ptr = (ptr ? @"foo" : @"bar");
}

id test24(void) {
  extern void test24_helper(void);
  return test24_helper(), (void*) 0;
}

// rdar://9400841
@interface Base
@property (assign) id content;
@end

@interface Foo : Base
-(void)test;
@end

@implementation Foo
-(void)test {
	super.content = 0;
}
@end

// <rdar://problem/9398437>
void test25(Class *classes) {
  Class *other_classes;
  test25(other_classes);
}

void test26(id y) {
  extern id test26_var1;
  __sync_swap(&test26_var1, 0, y); // expected-error {{cannot perform atomic operation on a pointer to type '__strong id': type has non-trivial ownership}}

  extern __unsafe_unretained id test26_var2;
  __sync_swap(&test26_var2, 0, y);
}

@interface Test26
- (id) init;
- (id) initWithInt: (int) x;
@end
@implementation Test26
- (id) init { return self; }
- (id) initWithInt: (int) x {
  [self init]; // expected-error {{the result of a delegate init call must be immediately returned or assigned to 'self'}}
  return self;
}
@end

// rdar://9525555
@interface  Test27 {
  __weak id _myProp1;
  id myProp2;
}
@property id x;
@property (readonly) id ro;
@property (readonly) id custom_ro;
@property int y;

@property (readonly) __weak id myProp1;
@property (readonly) id myProp2;
@property (readonly) __strong id myProp3;
@end

@implementation Test27
@synthesize x;
@synthesize ro;
@synthesize y;

@synthesize myProp1 = _myProp1;
@synthesize myProp2;
@synthesize myProp3;

-(id)custom_ro { return 0; }
@end

// rdar://9569264
@interface Test28
@property (nonatomic, assign) __strong id a; // expected-error {{unsafe_unretained property 'a' may not also be declared __strong}}
@end

@interface Test28 ()
@property (nonatomic, assign) __strong id b; // expected-error {{unsafe_unretained property 'b' may not also be declared __strong}}
@end

@implementation Test28
@synthesize a;
@synthesize b;
@end

// rdar://9573962
typedef struct Bark Bark;
@interface Test29
@property Bark* P;
@end

@implementation Test29
@synthesize P;
- (id)Meth { 
  Bark** f = &P; 
  return 0; 
}
@end

// rdar://9495837
@interface Test30
+ (id) new;
- (void)Meth;
@end

@implementation Test30
+ (id) new { return 0; }
- (void) Meth {
  __weak id x = [Test30 new]; // expected-warning {{assigning retained object to weak variable}}
  id __unsafe_unretained u = [Test30 new]; // expected-warning {{assigning retained object to unsafe_unretained variable}}
  id y = [Test30 new];
  x = [Test30 new]; // expected-warning {{assigning retained object to weak variable}}
  u = [Test30 new]; // expected-warning {{assigning retained object to unsafe_unretained variable}}
  y = [Test30 new];
}
@end

// rdar://9411838
@protocol PTest31 @end

int Test31() {
    Class cls;
    id ids;
    id<PTest31> pids;
    Class<PTest31> pcls;

    int i =  (ids->isa ? 1 : 0); // expected-error {{member reference base type 'id' is not a structure or union}}
    int j = (pids->isa ? 1 : 0); // expected-error {{member reference base type 'id<PTest31>' is not a structure or union}}
    int k = (pcls->isa ? i : j); // expected-error {{member reference base type 'Class<PTest31>' is not a structure or union}}
    return cls->isa ? i : j; // expected-error {{member reference base type 'Class' is not a structure or union}}
}

// rdar://9612030
@interface ITest32 {
@public
 id ivar;
}
@end

id Test32(__weak ITest32 *x) {
  __weak ITest32 *y;
  x->ivar = 0; // expected-error {{dereferencing a __weak pointer is not allowed}}
  return y ? y->ivar     // expected-error {{dereferencing a __weak pointer is not allowed}}
           : (*x).ivar;  // expected-error {{dereferencing a __weak pointer is not allowed}}
}

// rdar://9619861
extern int printf(const char*, ...);
typedef long intptr_t;

int Test33(id someid) {
  printf( "Hello%ld", (intptr_t)someid);
  return (int)someid;
}

// rdar://9636091
@interface I34
@property (nonatomic, retain) id newName __attribute__((ns_returns_not_retained)) ;

@property (nonatomic, retain) id newName1 __attribute__((ns_returns_not_retained)) ;
- (id) newName1 __attribute__((ns_returns_not_retained));

@property (nonatomic, retain) id newName2 __attribute__((ns_returns_not_retained)); // expected-note {{roperty declared here}}
- (id) newName2;   // expected-warning {{property declared as returning non-retained objects; getter returning retained objects}}
@end

@implementation I34
@synthesize newName;

@synthesize newName1;
- (id) newName1 { return 0; }

@synthesize newName2;
@end

void test35(void) {
  extern void test36_helper(id*);
  id x;
  __strong id *xp = 0;

  test36_helper(&x);
  test36_helper(xp); // expected-error {{passing address of non-local object to __autoreleasing parameter for write-back}}

  // rdar://problem/9665710
  __block id y;
  test36_helper(&y);
  ^{ test36_helper(&y); }();

  __strong int non_objc_type; // expected-warning {{'__strong' only applies to Objective-C object or block pointer types}} 
}

void test36(int first, ...) {
  // <rdar://problem/9758798>
  __builtin_va_list arglist;
  __builtin_va_start(arglist, first);
  id obj = __builtin_va_arg(arglist, id);
  __builtin_va_end(arglist);
}

@class Test37; // expected-note{{forward declaration of class here}}
void test37(Test37 *c) {
  for (id y in c) { // expected-error {{collection expression type 'Test37' is a forward declaration}}
    (void) y;
  }

  (void)sizeof(id*); // no error.
}

// rdar://problem/9887979
@interface Test38
@property int value;
@end
void test38() {
  extern Test38 *test38_helper(void);
  switch (test38_helper().value) {
  case 0:
  case 1:
    ;
  }
}

// rdar://10186536
@class NSColor;
void _NSCalc(NSColor* color, NSColor* bezelColors[]) __attribute__((unavailable("not available in automatic reference counting mode")));

void _NSCalcBeze(NSColor* color, NSColor* bezelColors[]); // expected-error {{must explicitly describe intended ownership of an object array parameter}}

// rdar://9970739
@interface RestaurantTableViewCell
- (void) restaurantLocation;
@end

@interface Radar9970739
- (void) Meth;
@end

@implementation Radar9970739
- (void) Meth { 
  RestaurantTableViewCell *cell;
  [cell restaurantLocatoin]; // expected-error {{no visible @interface for 'RestaurantTableViewCell' declares the selector 'restaurantLocatoin'}}
}
@end

// rdar://11814185
@interface Radar11814185
@property (nonatomic, weak)  Radar11814185* picker1;
+ alloc;
- init;
@end

@implementation Radar11814185

@synthesize picker1;

- (void)viewDidLoad
{
    picker1 = [[Radar11814185 alloc] init]; // expected-warning {{assigning retained object to weak variable; object will be released after assignment}}
    self.picker1 = [[Radar11814185 alloc] init]; // expected-warning {{assigning retained object to weak property; object will be released after assignment}}
}

+ alloc { return 0; }
- init { return 0; }
@end

// <rdar://problem/12569201>.  Warn on cases of initializing a weak variable
// with an Objective-C object literal.
void rdar12569201(id key, id value) {
    // Declarations.
    __weak id x = @"foo"; // no-warning
    __weak id y = @{ key : value }; // expected-warning {{assigning dictionary literal to a weak variable; object will be released after assignment}}
    __weak id z = @[ value ]; // expected-warning {{assigning array literal to a weak variable; object will be released after assignment}}
    __weak id b = ^() {}; // expected-warning {{assigning block literal to a weak variable; object will be released after assignment}}
    __weak id n = @42; // expected-warning {{assigning numeric literal to a weak variable; object will be released after assignment}}
    __weak id e = @(42); // expected-warning {{assigning numeric literal to a weak variable; object will be released after assignment}}
    __weak id m = @(41 + 1); // expected-warning {{assigning boxed expression to a weak variable; object will be released after assignment}}
    
    // Assignments.
    y = @{ key : value }; // expected-warning {{assigning dictionary literal to a weak variable; object will be released after assignment}}
    z = @[ value ]; // expected-warning {{assigning array literal to a weak variable; object will be released after assignment}}
    b = ^() {}; // expected-warning {{assigning block literal to a weak variable; object will be released after assignment}}
    n = @42; // expected-warning {{assigning numeric literal to a weak variable; object will be released after assignment}}
    e = @(42); // expected-warning {{assigning numeric literal to a weak variable; object will be released after assignment}}
    m = @(41 + 1); // expected-warning {{assigning boxed expression to a weak variable; object will be released after assignment}}
}

@interface C
- (void)method:(id[])objects; // expected-error{{must explicitly describe intended ownership of an object array parameter}}
@end

// rdar://13752880
@interface NSMutableArray : NSArray @end

typedef __strong NSMutableArray * PSNS;

void test(NSArray *x) {
  NSMutableArray *y = x; // expected-warning {{incompatible pointer types initializing 'NSMutableArray *' with an expression of type 'NSArray *'}}
  __strong NSMutableArray *y1 = x; // expected-warning {{incompatible pointer types initializing 'NSMutableArray *' with an expression of type 'NSArray *'}}
  PSNS y2 = x; // expected-warning {{incompatible pointer types initializing 'NSMutableArray *' with an expression of type 'NSArray *'}}
}

// rdar://15123684
@class NSString;

void foo(NSArray *array) {
  for (NSString *string in array) {
    for (string in @[@"blah", @"more blah", string]) { // expected-error {{selector element of type 'NSString *const __strong' cannot be a constant l-value}}
    }
  }
}

// rdar://16627903
extern void abort();
#define TKAssertEqual(a, b) do{\
    __typeof(a) a_res = (a);\
    __typeof(b) b_res = (b);\
    if ((a_res) != (b_res)) {\
        abort();\
    }\
}while(0)

int garf() {
  id object;
  TKAssertEqual(object, nil);
  TKAssertEqual(object, (id)nil);
}
