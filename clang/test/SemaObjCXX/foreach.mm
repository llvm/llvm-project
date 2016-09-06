// RUN: %clang_cc1 -fsyntax-only -fblocks -verify -std=c++11 %s
// rdar://9293227

@class NSArray;

void f(NSArray *a) {
    id keys;
    for (int i : a); // expected-error{{selector element type 'int' is not a valid object}} 
    for ((id)2 : a);  // expected-error {{for range declaration must declare a variable}}
    for (2 : a); // expected-error {{for range declaration must declare a variable}}
  
  for (id thisKey : keys);

  for (auto thisKey : keys) { } // expected-warning{{'auto' deduced as 'id' in declaration of 'thisKey'}}
}

template<typename Collection>
void ft(Collection col) {
  for (id x : col) { }
  for (auto x : col) { }
}

template void ft(NSArray *);

/* // rdar://9072298 */
@protocol NSObject @end

@interface NSObject <NSObject> {
    Class isa;
}
@end

typedef struct {
    unsigned long state;
    id *itemsPtr;
    unsigned long *mutationsPtr;
    unsigned long extra[5];
} NSFastEnumerationState;

@protocol NSFastEnumeration

- (unsigned long)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(unsigned long)len;

@end

int main ()
{
 NSObject<NSFastEnumeration>* collection = 0;
 for (id thing : collection) { }

 id array;
 for (int (^b)(void) : array) {
    if (b() == 10000) {
        return 1;
    }
 }
 return 0;
}

/* rdar://problem/11068137 */
@interface Test2
@property (assign) id prop;
@end
void test2(NSObject<NSFastEnumeration> *collection) {
  Test2 *obj;
  for (obj.prop : collection) { // expected-error {{for range declaration must declare a variable}}
  }
}

void testErrors(NSArray *array) {
  typedef int fn(int);

  for (fn x in array) { } // expected-error{{non-variable declaration in 'for' loop}}
}
