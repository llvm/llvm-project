// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://10904488

@interface Test
- (int)objectAtIndexedSubscript:(int)index; // expected-note {{method 'objectAtIndexedSubscript:' declared here}}
- (void)setObject:(int)object atIndexedSubscript:(int)index; // expected-note {{parameter of type 'int' is declared here}}
@end

@interface NSMutableDictionary
- (int)objectForKeyedSubscript:(id)key; // expected-note {{method 'objectForKeyedSubscript:' declared here}}
- (void)setObject:(int)object forKeyedSubscript:(id)key; // expected-note {{parameter of type 'int' is declared here}}
@end

int main(void) {
   Test *array;
   int i = array[10]; // expected-error {{method for accessing array element must have Objective-C object return type instead of 'int'}}
   array[2] = i;     // expected-error {{cannot assign to this array because assigning method's 2nd parameter of type 'int' is not an Objective-C pointer type}}

   NSMutableDictionary *dict;
   id key, val;
   val = dict[key]; // expected-error {{method for accessing dictionary element must have Objective-C object return type instead of 'int'}} \
                    // expected-error {{incompatible integer to pointer conversion assigning to 'id' from 'int'}}
   dict[key] = val; // expected-error {{method object parameter type 'int' is not object type}}
}

