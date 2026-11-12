// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -verify -Wno-objc-root-class %s

typedef long int NSUInteger;
#define nil 0
@class NSString;

@interface NSMutableArray

- (void)addObject:(id)object;
- (void)insertObject:(id)object atIndex:(NSUInteger)index;
- (void)replaceObjectAtIndex:(NSUInteger)index withObject:(id)object;
- (void)setObject:(id)object atIndexedSubscript:(NSUInteger)index;

@end

@interface NSMutableDictionary

- (void)setObject:(id)object forKey:(id)key;
- (void)setObject:(id)object forKeyedSubscript:(id)key;
- (void)setValue:(id)value forKey:(NSString *)key;

@end

@interface NSMutableSet

- (void)addObject:(id)object;

@end

@interface NSCountedSet : NSMutableSet

@end

@interface NSMutableOrderedSet

- (void)addObject:(id)object;
- (void)insertObject:(id)object atIndex:(NSUInteger)index;
- (void)setObject:(id)object atIndexedSubscript:(NSUInteger)index;
- (void)replaceObjectAtIndex:(NSUInteger)index withObject:(id)object;
- (void)setObject:(id)object atIndex:(NSUInteger)index;

@end

@interface SelfRefClass
{
  NSMutableArray *_array; // expected-note {{'_array' declared here}}
  NSMutableDictionary *_dictionary; // expected-note {{'_dictionary' declared here}}
  NSMutableSet *_set; // expected-note {{'_set' declared here}}
  NSCountedSet *_countedSet; // expected-note {{'_countedSet' declared here}}
  NSMutableOrderedSet *_orderedSet; // expected-note {{'_orderedSet' declared here}}
}
@end

@implementation SelfRefClass

- (void)check {
  [_array addObject:_array]; // expected-warning {{adding '_array' to '_array' might cause circular dependency in container}}
  [_dictionary setObject:_dictionary forKey:@"key"]; // expected-warning {{adding '_dictionary' to '_dictionary' might cause circular dependency in container}}
  [_set addObject:_set]; // expected-warning {{adding '_set' to '_set' might cause circular dependency in container}}
  [_countedSet addObject:_countedSet]; // expected-warning {{adding '_countedSet' to '_countedSet' might cause circular dependency in container}}
  [_orderedSet addObject:_orderedSet]; // expected-warning {{adding '_orderedSet' to '_orderedSet' might cause circular dependency in container}}
}

- (void)checkNSMutableArray:(NSMutableArray *)a { // expected-note {{'a' declared here}}
  [a addObject:a]; // expected-warning {{adding 'a' to 'a' might cause circular dependency in container}}
}

- (void)checkNSMutableDictionary:(NSMutableDictionary *)d { // expected-note {{'d' declared here}}
  [d setObject:d forKey:@"key"]; // expected-warning {{adding 'd' to 'd' might cause circular dependency in container}}
}

- (void)checkNSMutableSet:(NSMutableSet *)s { // expected-note {{'s' declared here}}
  [s addObject:s]; // expected-warning {{adding 's' to 's' might cause circular dependency in container}}
}

- (void)checkNSCountedSet:(NSCountedSet *)s { // expected-note {{'s' declared here}}
  [s addObject:s]; // expected-warning {{adding 's' to 's' might cause circular dependency in container}}
}

- (void)checkNSMutableOrderedSet:(NSMutableOrderedSet *)s { // expected-note {{'s' declared here}}
  [s addObject:s]; // expected-warning {{adding 's' to 's' might cause circular dependency in container}}
}

@end

void checkNSMutableArrayParam(NSMutableArray *a) { // expected-note {{'a' declared here}}
  [a addObject:a]; // expected-warning {{adding 'a' to 'a' might cause circular dependency in container}}
}

void checkNSMutableDictionaryParam(NSMutableDictionary *d) { // expected-note {{'d' declared here}}
  [d setObject:d forKey:@"key"]; // expected-warning {{adding 'd' to 'd' might cause circular dependency in container}}
}

void checkNSMutableSetParam(NSMutableSet *s) { // expected-note {{'s' declared here}}
  [s addObject:s]; // expected-warning {{adding 's' to 's' might cause circular dependency in container}}
}

void checkNSCountedSetParam(NSCountedSet *s) { // expected-note {{'s' declared here}}
  [s addObject:s]; // expected-warning {{adding 's' to 's' might cause circular dependency in container}}
}

void checkNSMutableOrderedSetParam(NSMutableOrderedSet *s) { // expected-note {{'s' declared here}}
  [s addObject:s]; // expected-warning {{adding 's' to 's' might cause circular dependency in container}}
}

void checkNSMutableArray(void) {
  NSMutableArray *a = nil; // expected-note 5 {{'a' declared here}} 5

  [a addObject:a]; // expected-warning {{adding 'a' to 'a' might cause circular dependency in container}}
  [a insertObject:a atIndex:0]; // expected-warning {{adding 'a' to 'a' might cause circular dependency in container}}
  [a replaceObjectAtIndex:0 withObject:a]; // expected-warning {{adding 'a' to 'a' might cause circular dependency in container}}
  [a setObject:a atIndexedSubscript:0]; // expected-warning {{adding 'a' to 'a' might cause circular dependency in container}}
  a[0] = a; // expected-warning {{adding 'a' to 'a' might cause circular dependency in container}}
}

void checkNSMutableDictionary(void) {
  NSMutableDictionary *d = nil; // expected-note 4 {{'d' declared here}}

  [d setObject:d forKey:@"key"]; // expected-warning {{adding 'd' to 'd' might cause circular dependency in container}}
  [d setObject:d forKeyedSubscript:@"key"]; // expected-warning {{adding 'd' to 'd' might cause circular dependency in container}}
  [d setValue:d forKey:@"key"]; // expected-warning {{adding 'd' to 'd' might cause circular dependency in container}}
  d[@"key"] = d; // expected-warning {{adding 'd' to 'd' might cause circular dependency in container}}
}

void checkNSMutableSet(void) {
  NSMutableSet *s = nil; // expected-note {{'s' declared here}}

  [s addObject:s]; // expected-warning {{adding 's' to 's' might cause circular dependency in container}}
}

void checkNSCountedSet(void) {
  NSCountedSet *s = nil; // expected-note {{'s' declared here}}

  [s addObject:s]; // expected-warning {{adding 's' to 's' might cause circular dependency in container}}
}

void checkNSMutableOrderedSet(void) {
  NSMutableOrderedSet *s = nil; // expected-note 5 {{'s' declared here}}

  [s addObject:s]; // expected-warning {{adding 's' to 's' might cause circular dependency in container}}
  [s insertObject:s atIndex:0]; // expected-warning {{adding 's' to 's' might cause circular dependency in container}}
  [s setObject:s atIndex:0]; // expected-warning {{adding 's' to 's' might cause circular dependency in container}}
  [s setObject:s atIndexedSubscript:0]; // expected-warning {{adding 's' to 's' might cause circular dependency in container}}
  [s replaceObjectAtIndex:0 withObject:s]; // expected-warning {{adding 's' to 's' might cause circular dependency in container}}
}

// Test subclassing

@interface FootableSet : NSMutableSet
@end

@implementation FootableSet
- (void)meth {
  [super addObject:self]; // expected-warning {{adding 'self' to 'super' might cause circular dependency in container}}
  [super addObject:nil]; // no-warning
  [self addObject:self]; // expected-warning {{adding 'self' to 'self' might cause circular dependency in container}}
}
@end

@interface FootableArray : NSMutableArray
@end

@implementation FootableArray
- (void)meth {
  [super addObject:self]; // expected-warning {{adding 'self' to 'super' might cause circular dependency in container}}
  [super addObject:nil]; // no-warning
  [self addObject:self]; // expected-warning {{adding 'self' to 'self' might cause circular dependency in container}}
}
@end

@interface FootableDictionary : NSMutableDictionary
@end

@implementation FootableDictionary
- (void)meth {
  [super setObject:self forKey:@"key"]; // expected-warning {{adding 'self' to 'super' might cause circular dependency in container}}
  [super setObject:nil forKey:@"key"]; // no-warning
  [self setObject:self forKey:@"key"]; // expected-warning {{adding 'self' to 'self' might cause circular dependency in container}}
}
@end


void subclassingNSMutableArray(void) {
  FootableArray *a = nil; // expected-note 5 {{'a' declared here}} 5

  [a addObject:a]; // expected-warning {{adding 'a' to 'a' might cause circular dependency in container}}
  [a insertObject:a atIndex:0]; // expected-warning {{adding 'a' to 'a' might cause circular dependency in container}}
  [a replaceObjectAtIndex:0 withObject:a]; // expected-warning {{adding 'a' to 'a' might cause circular dependency in container}}
  [a setObject:a atIndexedSubscript:0]; // expected-warning {{adding 'a' to 'a' might cause circular dependency in container}}
  a[0] = a; // expected-warning {{adding 'a' to 'a' might cause circular dependency in container}}
}

void subclassingNSMutableDictionary(void) {
  FootableDictionary *d = nil; // expected-note 4 {{'d' declared here}}

  [d setObject:d forKey:@"key"]; // expected-warning {{adding 'd' to 'd' might cause circular dependency in container}}
  [d setObject:d forKeyedSubscript:@"key"]; // expected-warning {{adding 'd' to 'd' might cause circular dependency in container}}
  [d setValue:d forKey:@"key"]; // expected-warning {{adding 'd' to 'd' might cause circular dependency in container}}
  d[@"key"] = d; // expected-warning {{adding 'd' to 'd' might cause circular dependency in container}}
}

void subclassingNSMutableSet(void) {
  FootableSet *s = nil; // expected-note {{'s' declared here}}

  [s addObject:s]; // expected-warning {{adding 's' to 's' might cause circular dependency in container}}
}

