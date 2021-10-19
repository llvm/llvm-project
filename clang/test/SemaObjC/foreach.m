/* RUN: %clang_cc1 -Wall -Wno-unused-but-set-variable -fsyntax-only -verify -std=c89 -pedantic %s
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
  for (id thisKey in keys); /* expected-warning {{unused variable 'thisKey'}} */
  for (id thisKey in keys); /* expected-warning {{unused variable 'thisKey'}} */
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
 for (id thing in collection) { } /* expected-warning {{unused variable 'thing'}} */

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

int cond();

void test3(NSObject<NSFastEnumeration> *a0, NSObject<NSFastEnumeration> *a1) {
  for (id i in a0) { /* expected-note 2 {{jump enters Objective-C fast enumeration loop}} */
    for (id j in a1) { /* expected-note 2 {{jump enters Objective-C fast enumeration loop}} */
      (void)i, (void)j;
label0:
      if (cond())
        goto label1;
    }
label1:
    if (cond())
      goto label0; /* expected-error {{cannot jump from this goto statement to its label}} */
    if (cond())
      goto label2;
  }

label2:
  if (cond())
    goto label0; /* expected-error {{cannot jump from this goto statement to its label}} */
  if (cond())
    goto label1; /* expected-error{{cannot jump from this goto statement to its label}} */
}
