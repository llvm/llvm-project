// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.Loops,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

#define nil ((id)0)

typedef unsigned long NSUInteger;
@protocol NSFastEnumeration
- (int)countByEnumeratingWithState:(void *)state objects:(id *)objects count:(unsigned)count;
- (void)protocolMethod;
@end

@interface NSObject
+ (instancetype)testObject;
@end

@interface NSEnumerator <NSFastEnumeration>
@end

@interface NSArray : NSObject <NSFastEnumeration>
- (NSUInteger)count;
- (NSEnumerator *)objectEnumerator;
@end

@interface NSDictionary : NSObject <NSFastEnumeration>
- (NSUInteger)count;
- (id)objectForKey:(id)key;
@end

@interface NSDictionary (SomeCategory)
- (void)categoryMethodOnNSDictionary;
@end

@interface NSMutableDictionary : NSDictionary
- (void)setObject:(id)obj forKey:(id)key;
@end

@interface NSMutableArray : NSArray
- (void)addObject:(id)obj;
@end

@interface NSSet : NSObject <NSFastEnumeration>
- (NSUInteger)count;
@end

@interface NSPointerArray : NSObject <NSFastEnumeration>
@end

@interface NSString : NSObject
@end

void test() {
  id x;
  for (x in [NSArray testObject])
    clang_analyzer_eval(x != nil); // expected-warning{{TRUE}}

  for (x in [NSMutableDictionary testObject])
    clang_analyzer_eval(x != nil); // expected-warning{{TRUE}}

  for (x in [NSSet testObject])
    clang_analyzer_eval(x != nil); // expected-warning{{TRUE}}

  for (x in [[NSArray testObject] objectEnumerator])
    clang_analyzer_eval(x != nil); // expected-warning{{TRUE}}

  for (x in [NSPointerArray testObject])
    clang_analyzer_eval(x != nil); // expected-warning{{UNKNOWN}}
}

void testWithVarInFor() {
  for (id x in [NSArray testObject])
    clang_analyzer_eval(x != nil); // expected-warning{{TRUE}}
  for (id x in [NSPointerArray testObject])
    clang_analyzer_eval(x != nil); // expected-warning{{UNKNOWN}}
}

void testNonNil(id a, id b) {
  clang_analyzer_eval(a != nil); // expected-warning{{UNKNOWN}}
  for (id x in a)
    clang_analyzer_eval(a != nil); // expected-warning{{TRUE}}

  if (b != nil)
    return;
  for (id x in b)
    *(volatile int *)0 = 1; // no-warning
  clang_analyzer_eval(b != nil); // expected-warning{{FALSE}}
}

void collectionIsEmpty(NSMutableDictionary *D){
  if ([D count] == 0) { // Count is zero.
    NSString *s = 0;
    for (NSString *key in D) {
      s = key;       // Loop is never entered.
    }
    clang_analyzer_eval(s == 0); //expected-warning{{TRUE}}
  }
}

void processCollection(NSMutableDictionary *D);
void collectionIsEmptyCollectionIsModified(NSMutableDictionary *D){
  if ([D count] == 0) {      // Count is zero.
    NSString *s = 0;
    processCollection(D);  // However, the collection has changed.
    for (NSString *key in D) {
      s = key;       // Loop might be entered.
    }
    clang_analyzer_eval(s == 0); //expected-warning{{FALSE}} //expected-warning{{TRUE}}
  }
}

int collectionIsEmptyNSSet(NSSet *S){
  if ([S count] == 2) { // Count is non-zero.
    int tapCounts[2];
    int i = 0;
    for (NSString *elem in S) {
      tapCounts[i]= 1;       // Loop is entered.
      i++;
    }
    return (tapCounts[0]); //no warning
  }
  return 0;
}

int collectionIsNotEmptyNSArray(NSArray *A) {
  int count = [A count];
  if (count > 0) {
    int i;
    int j;
    for (NSString *a in A) {
      i = 1;
      j++;
    }
    clang_analyzer_eval(i == 1); // expected-warning {{TRUE}}
  }
  return 0;
}

void onlySuppressExitAfterZeroIterations(NSMutableDictionary *D) {
  if (D.count > 0) {
    int *x;
    int i;
    for (NSString *key in D) {
      x = 0;
      i++;
    }
    // Test that this is reachable.
    int y = *x; // expected-warning {{Dereference of null pointer}}
    y++;
  }
}

void onlySuppressLoopExitAfterZeroIterations_WithContinue(NSMutableDictionary *D) {
  if (D.count > 0) {
    int *x;
    int i;
    for (NSString *key in D) {
      x = 0;
      i++;
      continue;
    }
    // Test that this is reachable.
    int y = *x; // expected-warning {{Dereference of null pointer}}
    y++;
  }
}

int* getPtr();
void onlySuppressLoopExitAfterZeroIterations_WithBreak(NSMutableDictionary *D) {
  if (D.count > 0) {
    int *x;
    int i;
    for (NSString *key in D) {
      x = 0;
      break;
      x = getPtr();
      i++;
    }
    int y = *x; // expected-warning {{Dereference of null pointer}}
    y++;
  }
}

int consistencyBetweenLoopsWhenCountIsUnconstrained(NSMutableDictionary *D,
                                                    int shouldUseCount) {
  // Test with or without an initial count.
  int count;
  if (shouldUseCount)
    count = [D count];

  int i;
  int j = 0;
  for (NSString *key in D) {
    i = 5;
    j++;
  }
  for (NSString *key in D)  {
    return i; // no-warning
  }
  return 0;
}

int consistencyBetweenLoopsWhenCountIsUnconstrained_dual(NSMutableDictionary *D,
                                                         int shouldUseCount) {
  int count;
  if (shouldUseCount)
    count = [D count];

  int i = 8;
  int j = 1;
  for (NSString *key in D) {
    i = 0;
    j++;
  }
  for (NSString *key in D)  {
    i = 5;
    j++;
  }
  return 5/i;
}

int consistencyCountThenLoop(NSArray *array) {
  if ([array count] == 0)
    return 0;

  int x;
  for (id y in array)
    x = 0;
  return x; // no-warning
}

int consistencyLoopThenCount(NSArray *array) {
  int x;
  for (id y in array)
    x = 0;

  if ([array count] == 0)
    return 0;

  return x; // no-warning
}

void nonMutatingMethodsDoNotInvalidateCountDictionary(NSMutableDictionary *dict,
                                                      NSMutableArray *other) {
  if ([dict count])
    return;

  for (id key in dict)
    clang_analyzer_eval(0); // no-warning

  (void)[dict objectForKey:@""];

  for (id key in dict)
    clang_analyzer_eval(0); // no-warning

  [dict categoryMethodOnNSDictionary];

  for (id key in dict)
    clang_analyzer_eval(0); // no-warning

  [dict setObject:@"" forKey:@""];

  for (id key in dict)
    clang_analyzer_eval(0); // expected-warning{{FALSE}}

  // Reset.
  if ([dict count])
    return;

  for (id key in dict)
    clang_analyzer_eval(0); // no-warning

  [other addObject:dict];

  for (id key in dict)
    clang_analyzer_eval(0); // expected-warning{{FALSE}}
}

void nonMutatingMethodsDoNotInvalidateCountArray(NSMutableArray *array,
                                                 NSMutableArray *other) {
  if ([array count])
    return;

  for (id key in array)
    clang_analyzer_eval(0); // no-warning

  (void)[array objectEnumerator];

  for (id key in array)
    clang_analyzer_eval(0); // no-warning

  [array addObject:@""];

  for (id key in array)
    clang_analyzer_eval(0); // expected-warning{{FALSE}}

  // Reset.
  if ([array count])
    return;

  for (id key in array)
    clang_analyzer_eval(0); // no-warning

  [other addObject:array];

  for (id key in array)
    clang_analyzer_eval(0); // expected-warning{{FALSE}}
}

void protocolMethods(NSMutableArray *array) {
  if ([array count])
    return;

  for (id key in array)
    clang_analyzer_eval(0); // no-warning

  NSArray *immutableArray = array;
  [immutableArray protocolMethod];

  for (id key in array)
    clang_analyzer_eval(0); // no-warning

  [array protocolMethod];

  for (id key in array)
    clang_analyzer_eval(0); // expected-warning{{FALSE}}
}
