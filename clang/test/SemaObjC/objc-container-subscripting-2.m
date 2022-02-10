// RUN: %clang_cc1  -fsyntax-only -verify %s

typedef unsigned int size_t;
@protocol P @end

@interface NSMutableArray
- (id)objectAtIndexedSubscript:(size_t)index;
- (void)setObject:(id)object atIndexedSubscript:(size_t)index;
@end

@interface NSMutableDictionary
- (id)objectForKeyedSubscript:(id)key;
- (void)setObject:(id)object forKeyedSubscript:(size_t)key;
@end

id func(void) {
  NSMutableArray *array;
  float f; 
  array[f] = array; // expected-error {{indexing expression is invalid because subscript type 'float' is not an integral or Objective-C pointer type}}
  return array[3.14]; // expected-error {{indexing expression is invalid because subscript type 'double' is not an integral or Objective-C pointer type}}
}

void test_unused(void) {
  NSMutableArray *array;
  array[10]; // expected-warning {{container access result unused - container access should not be used for side effects}} 

  NSMutableDictionary *dict;
  dict[array]; // expected-warning {{container access result unused - container access should not be used for side effects}}
}

void testQualifiedId(id<P> qualifiedId) {
  id object = qualifiedId[10];   // expected-error {{expected method to read array element not found on object of type 'id<P>'}}
  qualifiedId[10] = qualifiedId; // expected-error {{expected method to write array element not found on object of type 'id<P>'}}
}

void testUnqualifiedId(id unqualId) {
  id object = unqualId[10];
  unqualId[10] = object;
}

@protocol Subscriptable
- (id)objectAtIndexedSubscript:(size_t)index;
- (void)setObject:(id)object atIndexedSubscript:(size_t)index;
@end

void testValidQualifiedId(id<Subscriptable> qualifiedId) {
  id object = qualifiedId[10];
  qualifiedId[10] = object;
}
