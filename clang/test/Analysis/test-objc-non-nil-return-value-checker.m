// RUN: %clang_analyze_cc1 -analyzer-checker=osx.cocoa.NonNilReturnValue,debug.ExprInspection -verify %s

typedef unsigned int NSUInteger;
typedef signed char BOOL;

@protocol NSObject  - (BOOL)isEqual:(id)object; @end

@interface NSObject <NSObject> {}
+(id)alloc;
+(id)new;
-(id)init;
-(id)autorelease;
-(id)copy;
- (Class)class;
-(id)retain;
@end

@interface NSArray : NSObject
- (id)objectAtIndex:(unsigned long)index;
@end

@interface NSArray (NSExtendedArray)
- (id)objectAtIndexedSubscript:(NSUInteger)idx;
@end

@interface NSMutableArray : NSArray
- (void)replaceObjectAtIndex:(NSUInteger)index withObject:(id)anObject;
@end

@interface NSOrderedSet : NSObject
@end
@interface NSOrderedSet (NSOrderedSetCreation)
- (id)objectAtIndexedSubscript:(NSUInteger)idx;
@end

void clang_analyzer_eval(id);

void assumeThatNSArrayObjectAtIndexIsNeverNull(NSArray *A, NSUInteger i) {
  clang_analyzer_eval([A objectAtIndex: i]); // expected-warning {{TRUE}} 
  id subscriptObj = A[1];
  clang_analyzer_eval(subscriptObj); // expected-warning {{TRUE}} 
}

void assumeThatNSMutableArrayObjectAtIndexIsNeverNull(NSMutableArray *A, NSUInteger i) {
  clang_analyzer_eval([A objectAtIndex: i]); // expected-warning {{TRUE}} 
}

void assumeThatNSArrayObjectAtIndexedSubscriptIsNeverNull(NSOrderedSet *A, NSUInteger i) {
  clang_analyzer_eval(A[i]); // expected-warning {{TRUE}}
}