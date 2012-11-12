// RUN: %clang_cc1 -verify -Wno-deprecated-declarations -Wno-objc-root-class %s

@interface Unrelated
@end

@interface NSObject
+ (id)new;
+ (id)alloc;
- (NSObject *)init;

- (id)retain;  // expected-note{{instance method 'retain' is assumed to return an instance of its receiver type ('NSArray *')}}
- autorelease;

- (id)self;

- (id)copy;
- (id)mutableCopy;

// Do not infer when instance/class mismatches
- (id)newNotInferred;
- (id)alloc;
+ (id)initWithBlarg;
+ (id)self;

// Do not infer when the return types mismatch.
- (Unrelated *)initAsUnrelated;
@end

@interface NSString : NSObject
- (id)init;
- (id)initWithCString:(const char*)string;
@end

@interface NSArray : NSObject
- (unsigned)count;
@end

@interface NSBlah 
@end

@interface NSMutableArray : NSArray
@end

@interface NSBlah ()
+ (Unrelated *)newUnrelated;
@end

void test_inference() {
  // Inference based on method family
  __typeof__(([[NSString alloc] init])) *str = (NSString**)0;
  __typeof__(([[[[NSString new] self] retain] autorelease])) *str2 = (NSString **)0;
  __typeof__(([[NSString alloc] initWithCString:"blah"])) *str3 = (NSString**)0;

  // Not inferred
  __typeof__(([[NSString new] copy])) *id1 = (id*)0;

  // Not inferred due to instance/class mismatches
  __typeof__(([[NSString new] newNotInferred])) *id2 = (id*)0;
  __typeof__(([[NSString new] alloc])) *id3 = (id*)0;
  __typeof__(([NSString self])) *id4 = (id*)0;
  __typeof__(([NSString initWithBlarg])) *id5 = (id*)0;

  // Not inferred due to return type mismatch
  __typeof__(([[NSString alloc] initAsUnrelated])) *unrelated = (Unrelated**)0;
  __typeof__(([NSBlah newUnrelated])) *unrelated2 = (Unrelated**)0;

  
  NSArray *arr = [[NSMutableArray alloc] init];
  NSMutableArray *marr = [arr retain]; // expected-warning{{incompatible pointer types initializing 'NSMutableArray *' with an expression of type 'NSArray *'}}
}

@implementation NSBlah
+ (Unrelated *)newUnrelated {
  return (Unrelated *)0;
}
@end

@implementation NSBlah (Cat)
+ (Unrelated *)newUnrelated2 {
  return (Unrelated *)0;
}
@end

@interface A
- (id)initBlah; // expected-note 2{{overridden method is part of the 'init' method family}}
@end

@interface B : A
- (Unrelated *)initBlah; // expected-warning{{method is expected to return an instance of its class type 'B', but is declared to return 'Unrelated *'}}
@end

@interface C : A
@end

@implementation C
- (Unrelated *)initBlah {  // expected-warning{{method is expected to return an instance of its class type 'C', but is declared to return 'Unrelated *'}}
  return (Unrelated *)0;
}
@end

@interface D
+ (id)newBlarg; // expected-note{{overridden method is part of the 'new' method family}}
@end

@interface D ()
+ alloc; // expected-note{{overridden method is part of the 'alloc' method family}}
@end

@implementation D
+ (Unrelated *)newBlarg { // expected-warning{{method is expected to return an instance of its class type 'D', but is declared to return 'Unrelated *'}}
  return (Unrelated *)0;
}

+ (Unrelated *)alloc { // expected-warning{{method is expected to return an instance of its class type 'D', but is declared to return 'Unrelated *'}}
  return (Unrelated *)0;
}
@end

@protocol P1
- (id)initBlah; // expected-note{{overridden method is part of the 'init' method family}}
- (int)initBlarg;
@end

@protocol P2 <P1>
- (int)initBlah; // expected-warning{{protocol method is expected to return an instance of the implementing class, but is declared to return 'int'}}
- (int)initBlarg;
- (int)initBlech;
@end

@interface E
- init;
@end

@implementation E
- init {
  return self;
}
@end

@protocol P3
+ (NSString *)newString;
@end

@interface F<P3>
@end

@implementation F
+ (NSString *)newString { return @"blah"; }
@end

// <rdar://problem/9340699>
@interface G 
- (id)_ABC_init __attribute__((objc_method_family(init))); // expected-note {{method '_ABC_init' declared here}}
@end

@interface G (Additions)
- (id)_ABC_init2 __attribute__((objc_method_family(init)));
@end

@implementation G (Additions)
- (id)_ABC_init { // expected-warning {{category is implementing a method which will also be implemented by its primary class}}
  return 0;
}
- (id)_ABC_init2 {
  return 0;
}
- (id)_ABC_init3 {
  return 0;
}
@end

// PR12384
@interface Fail @end
@protocol X @end
@implementation Fail
- (id<X>) initWithX
{
  return (id)self; // expected-warning {{returning 'Fail *' from a function with incompatible result type 'id<X>'}}
}
@end

// <rdar://problem/11460990>

@interface WeirdNSString : NSString
- (id)initWithCString:(const char*)string, void *blah;
@end
