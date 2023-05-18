// RUN: %clang_cc1  -fsyntax-only -verify %s

typedef unsigned int size_t;
@protocol P @end

@interface NSMutableArray
- (id)objectAtIndexedSubscript:(double)index; // expected-note {{parameter of type 'double' is declared here}}
- (void)setObject:(id *)object atIndexedSubscript:(void *)index; // expected-note {{parameter of type 'void *' is declared here}} \
								 // expected-note {{parameter of type 'id *' is declared here}}
@end
@interface I @end

int main(void) {
  NSMutableArray<P> * array;
  id  oldObject = array[10]; // expected-error {{method index parameter type 'double' is not integral type}}
  array[3] = 0; // expected-error {{method index parameter type 'void *' is not integral type}} \
                // expected-error {{cannot assign to this array because assigning method's 2nd parameter of type 'id *' is not an Objective-C pointer type}}

  I* iarray;
  iarray[3] = 0; // expected-error {{expected method to write array element not found on object of type 'I *'}}
  I* p = iarray[4]; // expected-error {{expected method to read array element not found on object of type 'I *'}}

  oldObject = array[10]++; // expected-error {{illegal operation on Objective-C container subscripting}}
  oldObject = array[10]--; // expected-error {{illegal operation on Objective-C container subscripting}}
  oldObject = --array[10]; // expected-error {{illegal operation on Objective-C container subscripting}}
}

@interface NSMutableDictionary
- (id)objectForKeyedSubscript:(id*)key; // expected-note {{parameter of type 'id *' is declared here}}
- (void)setObject:(void*)object forKeyedSubscript:(id*)key; // expected-note {{parameter of type 'void *' is declared here}} \
                                                            // expected-note {{parameter of type 'id *' is declared here}}
@end
@class NSString;

void testDict(void) {
  NSMutableDictionary *dictionary;
  NSString *key;
  id newObject, oldObject;
  oldObject = dictionary[key];  // expected-error {{method key parameter type 'id *' is not object type}}
  dictionary[key] = newObject;  // expected-error {{method object parameter type 'void *' is not object type}} \
                                // expected-error {{method key parameter type 'id *' is not object type}}
}
