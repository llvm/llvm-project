// RUN: %clang_cc1 -verify %s

@interface Unrelated
@end

@interface NSObject
+ (id)new;
+ (id)alloc;
- (NSObject *)init;

- (id)retain;  // expected-note 2{{instance method 'retain' is assumed to return an instance of its receiver type ('NSArray *')}}
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
  marr = [arr retain]; // expected-warning{{incompatible pointer types assigning to 'NSMutableArray *' from 'NSArray *'}}
  arr = [marr retain];
}
