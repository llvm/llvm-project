// RUN: %clang_cc1 -arcmt-action=check -verify -triple x86_64-apple-darwin10 -fblocks -Werror %s

#if __has_feature(objc_arc)
#define NS_AUTOMATED_REFCOUNT_UNAVAILABLE __attribute__((unavailable("not available in automatic reference counting mode")))
#else
#define NS_AUTOMATED_REFCOUNT_UNAVAILABLE
#endif

typedef const void * CFTypeRef;
CFTypeRef CFBridgingRetain(id X);
id CFBridgingRelease(CFTypeRef);

typedef int BOOL;
typedef unsigned NSUInteger;

@protocol NSObject
- (id)retain NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (NSUInteger)retainCount NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (oneway void)release NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
- (id)autorelease NS_AUTOMATED_REFCOUNT_UNAVAILABLE;
@end

@interface NSObject <NSObject> {}
- (id)init;

+ (id)new;
+ (id)alloc;
- (void)dealloc;

- (void)finalize;

- (id)copy;
- (id)mutableCopy;
@end

typedef const struct __CFString * CFStringRef;
extern const CFStringRef kUTTypePlainText;
extern const CFStringRef kUTTypeRTF;
@class NSString;
@class A;

struct UnsafeS {
  A *__unsafe_unretained unsafeObj;
};

@interface A : NSObject
- (id)retain __attribute__((unavailable)); // expected-note {{'retain' has been explicitly marked unavailable here}}
- (id)retainCount __attribute__((unavailable)); // expected-note {{'retainCount' has been explicitly marked unavailable here}}
- (id)autorelease __attribute__((unavailable)); // expected-note 2 {{'autorelease' has been explicitly marked unavailable here}}
- (id)init;
- (oneway void)release;
- (void)dealloc;
-(void)test;
-(id)delegate;
@end

@implementation A
-(void)test {
  [super dealloc];
}
-(void)dealloc {
  [super dealloc];
}

- (id)retain { return self; } // expected-error {{ARC forbids implementation}}
- (id)retainCount { return self; } // expected-error {{ARC forbids implementation}}
- (id)autorelease { return self; } // expected-error {{ARC forbids implementation}}
- (oneway void)release { } // expected-error {{ARC forbids implementation}}

-(id)delegate { return self; }
@end

id global_foo;

void test1(A *a, BOOL b, struct UnsafeS *unsafeS) {
  [[a delegate] release]; // expected-error {{it is not safe to remove 'retain' message on the result of a 'delegate' message; the object that was passed to 'setDelegate:' may not be properly retained}} \
                          // expected-error {{ARC forbids explicit message send}}
  [a.delegate release]; // expected-error {{it is not safe to remove 'retain' message on the result of a 'delegate' message; the object that was passed to 'setDelegate:' may not be properly retained}} \
                        // expected-error {{ARC forbids explicit message send}}
  [unsafeS->unsafeObj retain]; // expected-error {{it is not safe to remove 'retain' message on an __unsafe_unretained type}} \
                               // expected-error {{ARC forbids explicit message send}} \
                               // expected-error {{'retain' is unavailable}}
  id foo = [unsafeS->unsafeObj retain]; // no warning.
  [global_foo retain]; // expected-error {{it is not safe to remove 'retain' message on a global variable}} \
                       // expected-error {{ARC forbids explicit message send}}
  [global_foo release]; // expected-error {{it is not safe to remove 'release' message on a global variable}} \
                        // expected-error {{ARC forbids explicit message send}}
  [a dealloc];
  [a retain];
  [a retainCount]; // expected-error {{ARC forbids explicit message send of 'retainCount'}} \
                   // expected-error {{'retainCount' is unavailable}}
  [a release];
  [a autorelease]; // expected-error {{it is not safe to remove an unused 'autorelease' message; its receiver may be destroyed immediately}} \
                   // expected-error {{ARC forbids explicit message send}} \
                   // expected-error {{'autorelease' is unavailable}}
  [a autorelease]; // expected-error {{it is not safe to remove an unused 'autorelease' message; its receiver may be destroyed immediately}} \
                   // expected-error {{ARC forbids explicit message send}} \
                   // expected-error {{'autorelease' is unavailable}}
  a = 0;

  CFStringRef cfstr;
  NSString *str = (NSString *)cfstr; // expected-error {{cast of C pointer type 'CFStringRef' (aka 'const struct __CFString *') to Objective-C pointer type 'NSString *' requires a bridged cast}} \
  // expected-note {{use __bridge to convert directly (no change in ownership)}} \
  // expected-note {{use CFBridgingRelease call to transfer ownership of a +1 'CFStringRef' (aka 'const struct __CFString *') into ARC}} \
  str = (NSString *)kUTTypePlainText;
  str = b ? kUTTypeRTF : kUTTypePlainText;
  str = (NSString *)(b ? kUTTypeRTF : kUTTypePlainText);
  str = (NSString *)a; // no change.

  SEL s = @selector(retain);  // expected-error {{ARC forbids use of 'retain' in a @selector}}
  s = @selector(release); // expected-error {{ARC forbids use of 'release' in a @selector}}
  s = @selector(autorelease); // expected-error {{ARC forbids use of 'autorelease' in a @selector}}
  s = @selector(dealloc); // expected-error {{ARC forbids use of 'dealloc' in a @selector}}

  static id __autoreleasing X1; // expected-error {{global variables cannot have __autoreleasing ownership}}
}

struct S {
  A* a;
};

@interface B
-(id)alloc;
- (id)initWithInt: (int) i;
@end

void rdar8861761(void) {
  B *o1 = [[B alloc] initWithInt:0];
  B *o2 = [B alloc];
  [o2 initWithInt:0];
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

  for (__strong id x in collection) { // expected-error {{use of undeclared identifier 'collection'}}
    x = 0;
  }
}
@end

void * cvt(id arg)
{
  void* voidp_val;
  (void)(int*)arg; // expected-error {{disallowed}}
  (void)(id)arg;
  (void)(__autoreleasing id*)arg; // expected-error {{disallowed}}
  (void)(id*)arg; // expected-error {{disallowed}}

  (void)(__autoreleasing id**)voidp_val;
  (void)(void*)voidp_val;
  (void)(void**)arg; // expected-error {{disallowed}}
  cvt((void*)arg); // expected-error 2 {{requires a bridged cast}} \
                   // expected-note 2 {{use __bridge to}} expected-note {{use CFBridgingRelease call}} expected-note {{use CFBridgingRetain call}}
  cvt(0);
  (void)(__strong id**)(0);
  return arg; // expected-error {{requires a bridged cast}} expected-note {{use __bridge}} expected-note {{use CFBridgingRetain call}}
}


void test12(id collection) {
  for (id x in collection) {
    x = 0;
  }

  for (__strong id x in collection) {
    x = 0;
  }
}

void test6(unsigned cond) {
  switch (cond) {
  case 0:
    ;
    id x; // expected-note {{jump bypasses initialization of __strong variable}}

  case 1: // expected-error {{cannot jump}}
    x = 0;
    break;
  }
}

@class Test8_incomplete;
@interface Test8_complete @end;
@interface Test8_super @end;
@interface Test8 : Test8_super
- (id) init00;
- (id) init01; // expected-note {{declaration in interface}}
- (id) init02;
- (id) init03; // covariance
- (id) init04; // covariance
- (id) init05;

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
- (Test8_super*) init31; // expected-note {{declaration in interface}}
- (Test8_super*) init32;
- (Test8_super*) init33;
- (Test8_super*) init34; // covariance
- (Test8_super*) init35;

- (Test8*) init40; // id exception to covariance
- (Test8*) init41; // expected-note {{declaration in interface}}
- (Test8*) init42;
- (Test8*) init43; // this should be a warning, but that's a general language thing, not an ARC thing
- (Test8*) init44;
- (Test8*) init45;

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

- (void) init01 {} // expected-error {{method was declared as an 'init' method, but its implementation doesn't match because its result type is not an object pointer}}
- (void) init11 {}
- (void) init21 {} // expected-error {{method was declared as an 'init' method, but its implementation doesn't match because its result type is not an object pointer}}
- (void) init31 {} // expected-error {{method was declared as an 'init' method, but its implementation doesn't match because its result type is not an object pointer}}
- (void) init41 {} // expected-error {{method was declared as an 'init' method, but its implementation doesn't match because its result type is not an object pointer}}
- (void) init51 {}

- (Test8_incomplete*) init02 { return 0; } // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_incomplete*) init12 { return 0; } // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_incomplete*) init22 { return 0; } // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_incomplete*) init32 { return 0; } // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_incomplete*) init42 { return 0; } // expected-error {{init methods must return a type related to the receiver type}}
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

- (Test8_complete*) init05 { return 0; } // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_complete*) init15 { return 0; } // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_complete*) init25 { return 0; } // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_complete*) init35 { return 0; } // expected-error {{init methods must return a type related to the receiver type}}
- (Test8_complete*) init45 { return 0; } // expected-error {{init methods must return a type related to the receiver type}}
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

void rdar9491791(int p) {
  switch (p) {
  case 3:;
    NSObject *o = [[NSObject alloc] init];
    [o release];
    break;
  default:
    break;
  }
}

#define RELEASE_MACRO(x) do { [x release]; } while(1)

void rdar9504750(id p) {
  RELEASE_MACRO(p); // expected-error {{ARC forbids explicit message send of 'release'}} 
}

@interface TestReadonlyProperty : NSObject
@property(assign,readonly) NSObject *value;
@end

@implementation TestReadonlyProperty
@synthesize value;
- (void)viewDidLoad {
  value = [NSObject new]; // expected-error {{assigning retained object}}
}
@end

@interface I9601437 {
  __unsafe_unretained id x;
}
-(void)Meth;
@end

@implementation I9601437
-(void)Meth {
  self->x = [NSObject new]; // expected-error {{assigning retained object}}
}
@end

@interface Test10 : NSObject {
  CFStringRef cfstr;
}
@property (retain) id prop;
-(void)foo;
@end

void test(Test10 *x) {
  x.prop = ^{ [x foo]; }; // expected-warning {{likely to lead to a retain cycle}} \
                          // expected-note {{retained by the captured object}}
}

@implementation Test10
-(void)foo {
  ^{
    NSString *str = (NSString *)cfstr; // expected-error {{cast of C pointer type 'CFStringRef' (aka 'const struct __CFString *') to Objective-C pointer type 'NSString *' requires a bridged cast}} \
    // expected-note {{use __bridge to convert directly (no change in ownership)}} \
    // expected-note {{use CFBridgingRelease call to transfer ownership of a +1 'CFStringRef' (aka 'const struct __CFString *') into ARC}}
  };
}
@end
