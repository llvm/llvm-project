/* RUN: %clang_cc1 -Wall -fsyntax-only -verify -std=c89 -pedantic %s
 */

@class NSArray;

void f(NSArray *a) {
    id keys;
    for (int i in a); /* expected-error{{selector element type 'int' is not a valid object}} */
    for ((id)2 in a); /* expected-error{{selector element is not a valid lvalue}} */
    for (2 in a); /* expected-error{{selector element is not a valid lvalue}} */
  
  /* This should be ok, 'thisKey' should be scoped to the loop in question,
   * and no diagnostics even in pedantic mode should happen.
   * rdar://6814674
   */
  for (id thisKey in keys);
  for (id thisKey in keys);
}

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
 NSObject<NSFastEnumeration>* collection = ((void*)0);
 for (id thing in collection) { }

 return 0;
}

/* rdar://problem/11068137 */
@interface Test2
@property (assign) id prop;
@end
void test2(NSObject<NSFastEnumeration> *collection) {
  Test2 *obj;
  for (obj.prop in collection) { /* expected-error {{selector element is not a valid lvalue}} */
  }
}
